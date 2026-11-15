"""Plot daily compounded returns from the paper replay and overlap experiment."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import pandas as pd


PAPER_COLORS = {
    "MZ": ("#087F76", "-"),
    "BZ": ("#903D83", (0, (4, 1.6))),
}
OVERLAP_COLORS = {
    "M": ("#556478", (0, (4, 2))),
    "CAP3": ("#1D4ED8", "-"),
    "RISK3": ("#087F76", "-"),
}


def compounded(daily, key, return_column, order):
    returns = daily.pivot(index="date", columns=key, values=return_column).sort_index()[order]
    if returns.isna().any().any():
        raise ValueError("All plotted series must share one complete trading calendar")
    initial = pd.DataFrame(0.0, index=[pd.Timestamp("2016-12-30")], columns=returns.columns)
    return pd.concat([initial, (1 + returns).cumprod() - 1])


def format_plot(title, subtitle, panel_titles, panels, styles, caption, output, font):
    colors = {"paper": "#F5F8FD", "ink": "#182941", "muted": "#526177", "grid": "#D8E1ED"}
    fig, axes = plt.subplots(
        1, len(panels), figsize=(12.4, 5.9), sharey=True, squeeze=False,
        facecolor=colors["paper"]
    )
    fig.subplots_adjust(left=.084, right=.972, bottom=.215, top=.745, wspace=.14)
    fig.text(.047, .924, title, fontsize=25, fontproperties=font, color=colors["ink"])
    fig.text(.048, .859, subtitle, fontsize=13, fontproperties=font, color=colors["muted"])

    low = min(panel.min().min() for panel in panels)
    high = max(panel.max().max() for panel in panels)
    span = high - low
    for ax, panel_title, panel in zip(axes.flat, panel_titles, panels):
        ax.set_facecolor("white")
        ax.set_title(panel_title, loc="left", pad=12, fontsize=17, fontproperties=font,
                     color=colors["ink"])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=colors["grid"], linewidth=.7)
        ax.axhline(0, color="#A0AEC1", linewidth=.9, zorder=1)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(colors["grid"])
        ax.tick_params(axis="both", length=0, pad=8, labelsize=11, colors=colors["muted"])
        ax.yaxis.set_major_locator(MultipleLocator(.10))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(pd.Timestamp("2016-12-23"), pd.Timestamp("2023-06-21"))
        ax.set_ylim(low - .045 * span, high + .045 * span)
        for label, series in panel.items():
            color, linestyle = styles[label]
            ax.plot(series.index, series, label=f"{label}  {series.iloc[-1]:+.2%}",
                    color=color, linestyle=linestyle, linewidth=1.7)
            ax.scatter(series.index[-1], series.iloc[-1], color=color, edgecolor="white",
                       linewidth=.7, s=20, zorder=5)

        legend = ax.legend(loc="upper left", bbox_to_anchor=(-.01, -.113), frameon=False,
                           ncols=len(panel.columns), fontsize=11.5,
                           handlelength=1.55, handletextpad=.46, columnspacing=1.24,
                           borderpad=.2, labelspacing=.48, labelcolor=colors["ink"])
        for text in legend.get_texts():
            text.set_color(colors["ink"])

    fig.text(.048, .037, caption, fontsize=10.5, fontproperties=font, color=colors["muted"])
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--output", default=Path("reports"), type=Path)
    parser.add_argument("--font", type=Path, help="Font file with Korean glyphs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    apple_font = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    font_path = args.font or (apple_font if apple_font.is_file() else None)
    if font_path is None:
        font_path = fm.findfont(fm.FontProperties(family=["Noto Sans CJK KR", "NanumGothic"]),
                                fallback_to_default=False)
    font = fm.FontProperties(fname=font_path)
    paper = pd.read_csv(args.root / "results/paper/daily.csv", parse_dates=["date"])
    overlap = pd.read_csv(args.root / "results/paper_overlap/daily.csv", parse_dates=["date"])

    format_plot(
        "선행논문 Z 전략 재현",
        "2017년 1월–2023년 5월  ·  누적수익률",
        ["비용 전", "비용 후"],
        [compounded(paper, "strategy", cost, list(PAPER_COLORS))
         for cost in ("gross_return", "net_return")],
        PAPER_COLORS,
        "M 가중 비중복 매칭  ·  B ADF p값 상위 250페어  ·  활성 거래일 비용 1%/252",
        args.output / "paper_reproduction.png",
        font,
    )
    format_plot(
        "중복 완화와 분산 제약",
        "2017년 1월–2023년 5월  ·  비용 후 누적수익률",
        ["Z 신호"],
        [compounded(overlap.loc[overlap.signal.eq("Z")], "model", "net_return",
                    list(OVERLAP_COLORS))],
        OVERLAP_COLORS,
        "M 가중 비중복  ·  CAP3 종목당 최대 3페어  ·  RISK3 최대 3페어와 학습 분산 상한",
        args.output / "overlap_risk.png",
        font,
    )


if __name__ == "__main__":
    main()
