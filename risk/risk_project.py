"""
Мини-проект: расчёт и мониторинг рыночного риска портфеля (VaR, ES, бэктест, стресс-тесты, лимиты).

Шаги:
  1. Данные: цены 5 факторов риска (симуляция; можно заменить на реальный CSV, см. load_prices)
  2. SQL: загрузка в SQLite + проверки качества данных (дубликаты, NULL, пропуски дат, выбросы)
  3. Доходности и портфель
  4. VaR / ES тремя методами + EWMA (скользящее окно)
  5. Бэктест: тест Купика, тест Кристофферсена, зоны светофора Basel
  6. Анализ чувствительности и стресс-тесты
  7. Мониторинг лимитов
  8. Графики и CSV-отчёты в папку output/
"""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

SEED = 42
CONF = 0.99            # уровень доверия
WINDOW = 250           # окно истории (торговых дней)
PORTFOLIO_VALUE = 100_000_000  # стоимость портфеля, $
N_SIM = 20_000         # число сценариев Монте-Карло
VAR_LIMIT = 2_600_000  # лимит на 1-дневный VaR 99%, $
WARN_LEVEL = 0.90      # «жёлтая зона»: утилизация лимита от 90%

ASSETS = ["KZ_BANK", "KZ_OIL", "US_EQUITY", "USDKZT", "KZ_BOND"]
WEIGHTS = np.array([0.25, 0.20, 0.30, 0.10, 0.15])


# ---------------------------------------------------------------- 1. ДАННЫЕ
def simulate_prices(n_days=1500, seed=SEED) -> pd.DataFrame:
    """Симуляция цен: толстые хвосты (t-распределение), кластеры волатильности (GARCH),
    корреляции и кризисный период (шок курса и акций). Это учебные данные, не реальные котировки."""
    rng = np.random.default_rng(seed)
    n = len(ASSETS)
    corr = np.array([
        [1.0, 0.5, 0.4, 0.3, 0.2],
        [0.5, 1.0, 0.4, 0.4, 0.1],
        [0.4, 0.4, 1.0, 0.2, 0.1],
        [0.3, 0.4, 0.2, 1.0, 0.3],
        [0.2, 0.1, 0.1, 0.3, 1.0],
    ])
    L = np.linalg.cholesky(corr)
    base_vol = np.array([0.015, 0.018, 0.011, 0.005, 0.003])  # дневная волатильность
    omega, alpha, beta = 0.05, 0.10, 0.85                      # GARCH(1,1) в относительных единицах
    nu = 5                                                      # степени свободы t-распределения
    h = np.ones(n)
    rets = np.zeros((n_days, n))
    for t in range(n_days):
        z = rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2))
        e = L @ z
        sigma = base_vol * np.sqrt(h)
        crisis = 1000 <= t < 1040
        if crisis:
            sigma = sigma * 2.5  # в кризис волатильность резко растёт
        r = sigma * e + 0.0003
        if crisis:  # и появляется отрицательный дрейф (тенге слабеет: USDKZT растёт)
            r += np.array([-0.012, -0.010, -0.008, 0.012, -0.002])
        rets[t] = r
        h = (1 - alpha - beta) * 1.0 + alpha * e**2 + beta * h
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=ASSETS)


def load_prices(csv_path: str | None = None) -> pd.DataFrame:
    """Если есть свой CSV (колонка date + колонки цен), используйте его вместо симуляции."""
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        return df
    return simulate_prices()


# ---------------------------------------------------------------- 2. SQL
def build_database(prices: pd.DataFrame) -> sqlite3.Connection:
    """Кладём цены в SQL-таблицу в «длинном» формате и намеренно портим данные,
    чтобы показать проверки качества."""
    long = (prices.reset_index().rename(columns={"index": "trade_date"})
            .melt(id_vars="trade_date", var_name="ticker", value_name="close_price"))
    long["trade_date"] = long["trade_date"].dt.strftime("%Y-%m-%d")

    rng = np.random.default_rng(7)
    # 1) пропуски цен
    idx = rng.choice(long.index, 6, replace=False)
    long.loc[idx, "close_price"] = np.nan
    # 2) дубликаты
    dups = long.sample(4, random_state=1)
    long = pd.concat([long, dups], ignore_index=True)
    # 3) выброс (опечатка в цене)
    long.loc[long.index[500], "close_price"] *= 10
    # 4) пропущенная дата целиком
    missing_day = long["trade_date"].unique()[300]
    long = long[long["trade_date"] != missing_day]

    con = sqlite3.connect(":memory:")
    long.to_sql("prices", con, index=False)
    return con


SQL_CHECKS = {
    "Дубликаты (дата+тикер)": """
        SELECT trade_date, ticker, COUNT(*) AS n FROM prices
        GROUP BY trade_date, ticker HAVING COUNT(*) > 1""",
    "Пропущенные цены (NULL)": """
        SELECT trade_date, ticker FROM prices WHERE close_price IS NULL""",
    "Даты, где записей меньше, чем тикеров": """
        SELECT trade_date, COUNT(DISTINCT ticker) AS n_tickers FROM prices
        GROUP BY trade_date HAVING COUNT(DISTINCT ticker) < (SELECT COUNT(DISTINCT ticker) FROM prices)""",
    "Пропущенные рабочие дни (нет даты в календаре)": """
        -- между соседними датами 2 дня (пропущен будний день) или 3+ дня не через выходные
        SELECT prev_date, trade_date FROM (
            SELECT trade_date, LAG(trade_date) OVER (ORDER BY trade_date) AS prev_date
            FROM (SELECT DISTINCT trade_date FROM prices))
        WHERE julianday(trade_date) - julianday(prev_date) = 2
           OR (julianday(trade_date) - julianday(prev_date) >= 3 AND strftime('%w', trade_date) <> '1')
           OR julianday(trade_date) - julianday(prev_date) > 3""",
    "Выбросы: дневное изменение > 25%": """
        SELECT * FROM (
            SELECT trade_date, ticker, close_price,
                   close_price * 1.0 / LAG(close_price) OVER (PARTITION BY ticker ORDER BY trade_date) - 1 AS ret
            FROM (SELECT DISTINCT trade_date, ticker, close_price FROM prices WHERE close_price IS NOT NULL))
        WHERE ABS(ret) > 0.25 ORDER BY trade_date""",
}


def run_quality_checks(con) -> dict:
    results = {}
    for name, q in SQL_CHECKS.items():
        results[name] = pd.read_sql(q, con)
    return results


def get_clean_prices(con) -> pd.DataFrame:
    """Чистим: убираем дубликаты, исправляем выбросы (>25% за день и сразу возврат), заполняем пропуски."""
    df = pd.read_sql("SELECT DISTINCT trade_date, ticker, close_price FROM prices", con,
                     parse_dates=["trade_date"])
    wide = df.pivot_table(index="trade_date", columns="ticker", values="close_price", aggfunc="first")[ASSETS]
    # выбросы: дневная доходность |r| > 25% -> считаем ошибкой, обнуляем и заполняем
    r = wide.pct_change().abs()
    wide = wide.mask(r > 0.25)
    # восстанавливаем полный календарь рабочих дней и заполняем пропуски предыдущим значением
    full = pd.bdate_range(wide.index.min(), wide.index.max())
    wide = wide.reindex(full).ffill()
    return wide


# ---------------------------------------------------------------- 3-4. VaR / ES
def var_es_hist(port_ret, conf=CONF):
    loss = -np.asarray(port_ret)
    var = np.percentile(loss, conf * 100)
    es = loss[loss >= var].mean()
    return var, es


def var_es_param(port_ret, conf=CONF):
    mu, sd = np.mean(port_ret), np.std(port_ret, ddof=1)
    z = stats.norm.ppf(conf)
    var = -mu + z * sd
    es = -mu + sd * stats.norm.pdf(z) / (1 - conf)
    return var, es


def var_es_mc(window_rets: np.ndarray, weights, conf=CONF, n_sim=N_SIM, rng=None):
    rng = rng or np.random.default_rng(0)
    mu = window_rets.mean(axis=0)
    cov = np.cov(window_rets, rowvar=False)
    L = np.linalg.cholesky(cov)
    sim = mu + rng.standard_normal((n_sim, len(mu))) @ L.T
    loss = -(sim @ weights)
    var = np.percentile(loss, conf * 100)
    return var, loss[loss >= var].mean()


def var_ewma(window_rets: np.ndarray, weights, conf=CONF, lam=0.94):
    w = lam ** np.arange(len(window_rets))[::-1]
    w /= w.sum()
    demeaned = window_rets - window_rets.mean(axis=0)
    cov = (demeaned.T * w) @ demeaned
    sd = np.sqrt(weights @ cov @ weights)
    return stats.norm.ppf(conf) * sd


def rolling_var(rets: pd.DataFrame, weights, window=WINDOW) -> pd.DataFrame:
    """Для каждого дня t считаем VaR по предыдущим `window` дням (без заглядывания в будущее)."""
    rng = np.random.default_rng(1)
    port = rets.values @ weights
    rows = []
    for t in range(window, len(rets)):
        w_rets = rets.values[t - window:t]
        w_port = port[t - window:t]
        h, h_es = var_es_hist(w_port)
        p, p_es = var_es_param(w_port)
        m, m_es = var_es_mc(w_rets, weights, rng=rng)
        e = var_ewma(w_rets, weights)
        rows.append((rets.index[t], port[t], h, h_es, p, p_es, m, m_es, e))
    cols = ["date", "pnl_ret", "VaR_hist", "ES_hist", "VaR_param", "ES_param", "VaR_mc", "ES_mc", "VaR_ewma"]
    return pd.DataFrame(rows, columns=cols).set_index("date")


# ---------------------------------------------------------------- 5. БЭКТЕСТ
def kupiec_test(n, x, p=1 - CONF):
    """H0: доля превышений = p. p-value < 0.05 -> модель отвергаем."""
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        pi = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen_independence(exc: np.ndarray):
    """H0: превышения не группируются (независимы). Низкий p-value -> превышения идут «кучами»."""
    exc = exc.astype(int)
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(exc[:-1], exc[1:]):
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1
    if min(n00 + n01, n10 + n11) == 0 or n01 == 0:
        return np.nan, np.nan
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(p, a, b):
        return (a * np.log(1 - p) if a else 0) + (b * np.log(p) if b and p > 0 else 0)

    l0 = ll(pi, n00 + n10, n01 + n11)
    l1 = ll(pi01, n00, n01) + ll(pi11, n10, n11)
    lr = -2 * (l0 - l1)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def basel_zone(x_250):
    """Зоны светофора Basel для 250 дней и VaR 99%."""
    if x_250 <= 4: return "green"
    if x_250 <= 9: return "yellow"
    return "red"


def verdict(x, n, p_kupiec, p_christoffersen):
    """Словесный вывод по двум тестам."""
    expected = n * (1 - CONF)
    parts = []
    if p_kupiec < 0.05:
        parts.append("частота превышений не та: риск " + ("недооценён" if x > expected else "переоценён"))
    else:
        parts.append("частота превышений в норме")
    if not np.isnan(p_christoffersen) and p_christoffersen < 0.05:
        parts.append("превышения идут кучами (модель медленно реагирует на кризис)")
    return "; ".join(parts)


def backtest_table(rv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    loss = -rv["pnl_ret"] * PORTFOLIO_VALUE
    for m in ["VaR_hist", "VaR_param", "VaR_mc", "VaR_ewma"]:
        exc = (loss > rv[m] * PORTFOLIO_VALUE).values
        n, x = len(exc), int(exc.sum())
        lr_k, p_k = kupiec_test(n, x)
        lr_c, p_c = christoffersen_independence(exc)
        last250 = int(exc[-250:].sum())
        rows.append({
            "Метод": m, "Дней": n, "Превышений": x,
            "Ожидалось": round(n * (1 - CONF), 1),
            "Доля, %": round(100 * x / n, 2),
            "Kupiec p-value": round(p_k, 4),
            "Christoffersen p-value": None if np.isnan(p_c) else round(p_c, 4),
            "Превышений за последние 250д": last250,
            "Basel зона": basel_zone(last250),
            "Вердикт": verdict(x, n, p_k, p_c),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 6. ЧУВСТВИТЕЛЬНОСТЬ И СТРЕСС
def sensitivity(weights):
    """Сколько $ потеряем/заработаем, если фактор изменится на -1%."""
    return pd.Series(-0.01 * weights * PORTFOLIO_VALUE, index=ASSETS, name="PnL при -1% фактора, $")


# Шоки заданы как изменение цены/курса фактора. Позиция USDKZT = длинная позиция в долларах:
# рост USDKZT (ослабление тенге) даёт прибыль, это естественный хедж при девальвации.
# Так же она считается и в симуляции доходностей, поэтому модули согласованы.
STRESS_SCENARIOS = {
    "Девальвация тенге (USDKZT +25%, акции KZ -15%, нефть -10%)":
        dict(KZ_BANK=-0.15, KZ_OIL=-0.10, US_EQUITY=-0.03, USDKZT=0.25, KZ_BOND=-0.05),
    "Глобальный кризис (акции -35%, нефть -40%, USDKZT +10%)":
        dict(KZ_BANK=-0.30, KZ_OIL=-0.40, US_EQUITY=-0.35, USDKZT=0.10, KZ_BOND=-0.08),
    "Рост ставок (облигации -10%, акции -8%)":
        dict(KZ_BANK=-0.08, KZ_OIL=-0.03, US_EQUITY=-0.08, USDKZT=0.02, KZ_BOND=-0.10),
    "Падение нефти (нефть -50%, банки -15%, USDKZT +15%)":
        dict(KZ_BANK=-0.15, KZ_OIL=-0.50, US_EQUITY=-0.05, USDKZT=0.15, KZ_BOND=-0.04),
}


def stress_test(weights) -> pd.DataFrame:
    rows = []
    for name, shock in STRESS_SCENARIOS.items():
        s = pd.Series(shock)[ASSETS]
        pnl = float((s.values * weights).sum() * PORTFOLIO_VALUE)
        rows.append({"Сценарий": name, "PnL, $": round(pnl), "PnL, % портфеля": round(pnl / PORTFOLIO_VALUE * 100, 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 7. ЛИМИТЫ
def monitor_limits(rv: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "VaR_$": rv["VaR_hist"] * PORTFOLIO_VALUE,
        "Лимит_$": VAR_LIMIT,
    })
    df["Утилизация_%"] = (df["VaR_$"] / df["Лимит_$"] * 100).round(1)
    df["Статус"] = np.where(df["Утилизация_%"] >= 100, "ПРЕВЫШЕНИЕ",
                    np.where(df["Утилизация_%"] >= WARN_LEVEL * 100, f"ВНИМАНИЕ (>={int(WARN_LEVEL*100)}%)", "OK"))
    return df


def breach_episodes(lim: pd.DataFrame) -> pd.DataFrame:
    """Склеиваем подряд идущие дни превышения в «эпизоды»."""
    b = lim["Статус"].eq("ПРЕВЫШЕНИЕ")
    grp = (b != b.shift()).cumsum()[b]
    rows = [{"Начало": g.index[0].date(), "Конец": g.index[-1].date(), "Дней": len(g),
             "Макс. VaR, $": round(g["VaR_$"].max()), "Макс. утилизация, %": g["Утилизация_%"].max()}
            for _, g in lim[b].groupby(grp)]
    return pd.DataFrame(rows)


def explain_breaches(rets: pd.DataFrame, episode_start, window=WINDOW) -> pd.DataFrame:
    """Причина превышения: какие факторы дали больше всего риска в окне на дату начала эпизода
    и во сколько раз выросла их волатильность по сравнению с предыдущим окном."""
    i = rets.index.get_loc(pd.Timestamp(episode_start))
    w_now = rets.iloc[i - window:i]
    w_prev = rets.iloc[max(i - 2 * window, 0):i - window]
    vol_now, vol_prev = w_now.std(), w_prev.std()
    cov = np.cov(w_now.values, rowvar=False)
    sd_p = np.sqrt(WEIGHTS @ cov @ WEIGHTS)
    contrib = WEIGHTS * (cov @ WEIGHTS) / sd_p ** 2      # доля фактора в дисперсии портфеля
    return pd.DataFrame({
        "Вклад в риск, %": (contrib * 100).round(1),
        "Волатильность сейчас, %/день": (vol_now * 100).round(2),
        "Волатильность прошлое окно, %/день": (vol_prev * 100).round(2),
        "Рост волатильности, x": (vol_now / vol_prev).round(2),
    })


# ---------------------------------------------------------------- 8. ГРАФИКИ
def make_charts(rv, stress_df, sens):
    loss = -rv["pnl_ret"] * PORTFOLIO_VALUE / 1e6

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(loss.index, loss, color="#9aa0a6", lw=0.7, label="Дневной убыток, млн $")
    ax.plot(rv.index, rv["VaR_hist"] * PORTFOLIO_VALUE / 1e6, color="#1a73e8", lw=1.5, label="VaR 99% (исторический)")
    ax.plot(rv.index, rv["ES_hist"] * PORTFOLIO_VALUE / 1e6, color="#d93025", lw=1.2, ls="--", label="ES 99%")
    exc = loss > rv["VaR_hist"] * PORTFOLIO_VALUE / 1e6
    ax.scatter(loss.index[exc], loss[exc], color="#d93025", s=22, zorder=5, label=f"Превышения VaR ({exc.sum()})")
    ax.axhline(VAR_LIMIT / 1e6, color="#e37400", ls=":", label="Лимит VaR")
    ax.set_title("Убыток портфеля, VaR, ES и превышения"); ax.set_ylabel("млн $"); ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT / "1_var_es_backtest.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for m, c in [("VaR_hist", "#1a73e8"), ("VaR_param", "#188038"), ("VaR_mc", "#e37400"), ("VaR_ewma", "#a142f4")]:
        ax.plot(rv.index, rv[m] * PORTFOLIO_VALUE / 1e6, lw=1.2, label=m, color=c)
    ax.set_title("Сравнение методов VaR 99%"); ax.set_ylabel("млн $"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "2_methods_comparison.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = [s.split(" (")[0] for s in stress_df["Сценарий"]]
    ax.barh(names, stress_df["PnL, $"] / 1e6, color="#d93025")
    ax.axvline(-VAR_LIMIT / 1e6, color="#e37400", ls=":", label="Лимит VaR (для масштаба)")
    ax.set_title("Стресс-тесты: убыток портфеля, млн $"); ax.legend(); ax.grid(alpha=.25, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "3_stress_tests.png", dpi=140); plt.close(fig)

    r = -rv["pnl_ret"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(r, bins=60, density=True, color="#9aa0a6", alpha=.8, label="Эмпирические убытки")
    xs = np.linspace(r.min(), r.max(), 300)
    ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), color="#1a73e8", label="Нормальное распределение")
    ax.set_title("Толстые хвосты: почему параметрический VaR занижает риск"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "4_fat_tails.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- MAIN
def main():
    print("=" * 70, "\n1-2. ДАННЫЕ И SQL-ПРОВЕРКИ\n", "=" * 70, sep="")
    prices_raw = load_prices()
    con = build_database(prices_raw)
    checks = run_quality_checks(con)
    for name, df in checks.items():
        print(f"\n[{name}] найдено строк: {len(df)}")
        if len(df):
            print(df.head(5).to_string(index=False))
    prices = get_clean_prices(con)
    print(f"\nПосле очистки: {prices.shape[0]} дней x {prices.shape[1]} факторов, NULL: {int(prices.isna().sum().sum())}")

    print("\n" + "=" * 70, "\n3. ДОХОДНОСТИ И ПОРТФЕЛЬ\n", "=" * 70, sep="")
    rets = np.log(prices / prices.shift(1)).dropna()
    port = rets.values @ WEIGHTS
    print("Веса:", {a: float(w) for a, w in zip(ASSETS, WEIGHTS)})
    print(f"Дневная волатильность портфеля: {port.std()*100:.2f}%, эксцесс: {stats.kurtosis(port):.2f} (у нормального = 0)")
    print(f"Тест Харке-Бера на нормальность: p = {stats.jarque_bera(port).pvalue:.2e} (малое p -> не нормальное, хвосты толстые)")

    print("\n" + "=" * 70, "\n4. VaR / ES НА ПОСЛЕДНИЙ ДЕНЬ\n", "=" * 70, sep="")
    rv = rolling_var(rets, WEIGHTS)
    last = rv.iloc[-1]
    summary = pd.DataFrame({
        "VaR 99%, $": [last.VaR_hist, last.VaR_param, last.VaR_mc, last.VaR_ewma],
        "ES 99%, $": [last.ES_hist, last.ES_param, last.ES_mc, np.nan],
    }, index=["Исторический", "Параметрический", "Монте-Карло", "EWMA (параметр.)"]) * PORTFOLIO_VALUE
    print(summary.round(0).to_string())
    print("\nES больше VaR, потому что ES усредняет убытки за порогом VaR.")

    print("\n" + "=" * 70, "\n5. БЭКТЕСТ\n", "=" * 70, sep="")
    bt = backtest_table(rv)
    print(bt.to_string(index=False))

    print("\n" + "=" * 70, "\n6. ЧУВСТВИТЕЛЬНОСТЬ И СТРЕСС-ТЕСТЫ\n", "=" * 70, sep="")
    sens = sensitivity(WEIGHTS)
    print(sens.round(0).to_string())
    stress_df = stress_test(WEIGHTS)
    print()
    print(stress_df.to_string(index=False))

    print("\n" + "=" * 70, "\n7. МОНИТОРИНГ ЛИМИТОВ\n", "=" * 70, sep="")
    lim = monitor_limits(rv)
    print(lim["Статус"].value_counts().to_string())
    episodes = breach_episodes(lim)
    if len(episodes):
        print("\nЭпизоды превышения лимита:")
        print(episodes.to_string(index=False))
        first = episodes.iloc[0]["Начало"]
        print(f"\nАнализ причины первого эпизода ({first}):")
        print(explain_breaches(rets, first).to_string())
        print("Читать так: если «Рост волатильности» заметно >1 у факторов с большим вкладом, причина в рынке;"
              " если вклад одного фактора сильно выше его веса, причина в концентрации позиции.")
    else:
        print("\nПревышений лимита нет.")

    print("\n" + "=" * 70, "\n8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ\n", "=" * 70, sep="")
    rv.to_csv(OUT / "rolling_var.csv")
    bt.to_csv(OUT / "backtest_summary.csv", index=False)
    stress_df.to_csv(OUT / "stress_tests.csv", index=False)
    lim.to_csv(OUT / "limit_monitoring.csv")
    summary.to_csv(OUT / "var_es_last_day.csv")
    make_charts(rv, stress_df, sens)
    print("Файлы сохранены в", OUT)


if __name__ == "__main__":
    main()
