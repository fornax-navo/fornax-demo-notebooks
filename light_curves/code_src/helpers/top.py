import re
from pathlib import Path

import attrs
import dateutil.parser
import matplotlib as mpl
import matplotlib.figure
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter

COLORS = tuple(
    mpl.colormaps["tab10"].colors[:5] + mpl.colormaps["tab10"].colors[9:] + mpl.colormaps["tab20b"].colors[::4]
)


def load_top_output(
    toptxt_file: str | Path = "top.txt",
    *,
    run_id: str = str(),
    pid_names: dict[int, str] = dict(),
    scrape_logs_for_pid_names: bool = True,
    toptxt_dir: Path | None = None,
) -> "_TopLog":
    """Parse `top` output from a text file and return it as a `_TopLog`.

    Parameters
    ----------
    toptxt_file : str or Path
        Name or path to the file containing `top` output. If `toptxt_dir` is not None, this should be
        relative to it. Otherwise, this may be an absolute path or relative to the current working directory.
    run_id : str (optional)
        ID for this output. Used to label figures.
    pid_names : dict[int, str] (optional)
        Mapping of PID numbers to names.
    scrape_logs_for_pid_names : bool
        Whether to search through the files in `toptxt_dir` looking for PID names.
    toptxt_dir: Path (optional)
        Path to the directory containing `toptxt_file`. If None, this will be the parent of `toptxt_file`.

    Returns
    -------
    TopLog
        The `top` output loaded as a `TopLog`.
    """
    toplog = _TopLog._from_txt_file(
        toptxt_file,
        toptxt_dir=toptxt_dir,
        run_id=run_id,
        pid_names=pid_names,
        scrape_logs_for_pid_names=scrape_logs_for_pid_names,
    )
    return toplog


@attrs.define(kw_only=True)
class _TopLog:
    file_path: Path = attrs.field(converter=Path)
    run_id: str = attrs.field(factory=str)
    summary_df: pd.DataFrame | None = attrs.field(default=None)
    pids_df: pd.DataFrame | None = attrs.field(default=None)
    pid_names: dict = attrs.field(factory=dict)
    pid_jobs: dict = attrs.field(factory=dict)
    pid_zorders: dict = attrs.field(factory=dict)
    pid_starts: dict = attrs.field(factory=dict)
    pid_ends: dict = attrs.field(factory=dict)
    job_colors: dict = attrs.field(factory=dict)
    time_tags: dict[str, "_TimeTag"] = attrs.field(factory=dict)
    default_color: str = attrs.field(default="0.75")

    @staticmethod
    def _scrape_logs_for_pid_names(logs_dir: Path | str) -> dict[int, str]:
        """Map PIDs to their names by scraping files in logs_dir."""
        pid_names = {}
        for logfile in Path(logs_dir).iterdir():
            if not logfile.is_file() or logfile.name.endswith(".sh.log"):
                continue
            pid_names.update(_TopLog._regex_log_for_pid_names(logfile))
        return pid_names

    @staticmethod
    def _regex_log_for_pid_names(logfile: Path) -> dict[int, str]:
        """Scrape the logfile for PIDs and their associated job names."""
        pid_names = {}
        pid_re = r"\[pid=([0-9]+)\]"
        archive_re = r"archive=([a-zA-Z0-9_-]+)"
        build_re = r"build=([a-zA-Z0-9_-]+)"

        with open(logfile) as fin:
            for line in fin:
                pid_match = re.search(pid_re, line)
                if not pid_match:
                    continue
                pid = int(pid_match.group(1))

                name_match = re.search(archive_re, line) or re.search(build_re, line)
                try:
                    pid_names[pid] = name_match.group(1)
                except AttributeError:
                    if "worker" not in line:
                        raise NotImplementedError(f"unable to identify a name for [pid={pid}]")
                    pid_names[pid] = logfile.stem + "-worker"

        return pid_names

    @classmethod
    def _from_txt_file(
        cls,
        toptxt_file: str | Path = "top.txt",
        *,
        run_id: str = str(),
        pid_names: dict[int, str] = dict(),
        scrape_logs_for_pid_names: bool = True,
        toptxt_dir: Path | None = None,
    ) -> "_TopLog":
        file_path = toptxt_dir / toptxt_file if toptxt_dir else toptxt_file
        toplog = cls(file_path=file_path, run_id=run_id)

        toplog.summary_df, toplog.pids_df = toplog._create_dfs(toplog._parse_toptxt())

        # pid and job names
        logs_pid_names = cls._scrape_logs_for_pid_names(toplog.file_path.parent) if scrape_logs_for_pid_names else {}
        logs_pid_names.update(**pid_names)
        logs_pid_names.update({pid: "" for pid in toplog.pids_df.PID.unique() if pid not in logs_pid_names})
        toplog.pid_names = {pid: logs_pid_names[pid] for pid in sorted(logs_pid_names)}
        toplog.pid_jobs = {pid: name.split("-")[0] for pid, name in toplog.pid_names.items()}
        toplog.pids_df.insert(0, "pid_name", toplog.pids_df.PID.map(toplog.pid_names))
        toplog.pids_df.insert(0, "job_name", toplog.pids_df.PID.map(toplog.pid_jobs))

        # pid_zorders
        pidgrps = toplog.pids_df.reset_index().groupby("PID")
        start, end = pidgrps.time.min(), pidgrps.time.max()
        ordered_pids = (end - start).sort_values(ascending=False).index.unique().tolist()
        toplog.pid_zorders = {pid: ordered_pids.index(pid) for pid in pidgrps.groups.keys()}
        toplog.pid_starts, toplog.pid_ends = start.to_dict(), end.to_dict()

        # job_colors
        colors = (color for color in COLORS)
        toplog.job_colors = {job: next(colors) for job in toplog.pids_df.job_name.unique() if job != ""}

        toplog.time_tags = {}
        if "time_tag" in toplog.summary_df.columns:
            time_tags = toplog.summary_df.query("time_tag != ''").time_tag.items()
            toplog.time_tags = {tag: _TimeTag(name=tag, time=time, color=next(colors)) for time, tag in time_tags}
            del toplog.summary_df["time_tag"]

        return toplog

    def _parse_toptxt(self):
        line_batch, regexed_batches = [], []
        with open(self.file_path, "r") as fin:
            for line in fin:
                if line.startswith("----") and len(line_batch) > 0:
                    regexed_batches.append(self._regex_top_line_batch(line_batch))
                    line_batch = []
                line_batch.append(line.removeprefix("----").rstrip("\n").strip())
            regexed_batches.append(self._regex_top_line_batch(line_batch))
        return regexed_batches

    def _create_dfs(self, regexed_batches):
        summary_df, pids_df = [pd.DataFrame(dicts) for dicts in zip(*regexed_batches)]

        pids_df = pids_df.explode(pids_df.columns.to_list(), ignore_index=True)

        for df in [summary_df, pids_df]:
            df["delta_time"] = df.time - df.time.min()
            df.dropna(axis=0, how="all", inplace=True)
            df.dropna(axis=1, how="all", inplace=True)

        if "time_tag" in summary_df.columns:
            summary_df["time_tag"] = summary_df["time_tag"].fillna("")

        for c in ["VIRT", "RES", "SHR"]:
            units = pids_df[c].str[-1].unique()
            if len(units) > 1:
                continue
            pids_df[c] = pids_df[c].str.strip(units[0]).astype(float)
            pids_df = pids_df.rename(columns={c: c + units[0]})
        dtype_map = {**{c: int for c in ["PID", "PR", "NI"]}, **{c: float for c in ["%CPU", "%MEM"]}}
        pids_df = pids_df.astype({c: dtype_map.get(c) for c in pids_df.columns if c in dtype_map})

        summary_df = summary_df.sort_values("time").set_index("time")
        pids_df = pids_df.sort_values(["time", "PID"]).set_index("time")
        return summary_df, pids_df

    def _regex_top_line_batch(self, line_batch: list[str]) -> tuple[dict, dict]:
        """Parse a batch of lines representing one dump of top."""
        if len(line_batch) < 7:
            # print(f"Warning: Skipping incomplete line batch. {line_batch}")
            return dict(), dict()

        # line 0
        batch_tag = line_batch[0]

        # line 1
        # date_format = "%Y/%m/%d %H:%M:%S %Z"
        # batch_time = datetime.strptime(line_batch[1], date_format)
        batch_time = dateutil.parser.parse(line_batch[1])

        # line 2
        load_match = re.search(r"(load average: )(.+)$", line_batch[2])
        load_avg = tuple(float(num) for num in load_match.group(2).split(", "))

        # lines 5-6
        ram_units = re.search(r"^(.+)( Mem )", line_batch[5]).group(1)
        mem_total = float(re.search(r"([0-9]+\.[0-9]+)( total)", line_batch[5]).group(1))
        mem_free = float(re.search(r"([0-9]+\.[0-9]+)( free)", line_batch[5]).group(1))
        mem_used = float(re.search(r"([0-9]+\.[0-9]+)( used)", line_batch[5]).group(1))
        mem_avail = float(re.search(r"([0-9]+\.[0-9]+)( avail Mem)$", line_batch[6]).group(1))

        # summary df
        load_names = [f"load_avg_{t}" for t in ["1m", "5m", "15m"]]
        mem_names = [f"{s}_{ram_units}" for s in ["total", "free", "used", "avail"]]
        columns = [*load_names, *mem_names, "time"]
        data = [*load_avg, mem_total, mem_free, mem_used, mem_avail, batch_time]
        if batch_tag:
            columns, data = ["time_tag"] + columns, [batch_tag] + data
        summary_dict = dict(zip(columns, data))
        if len(line_batch) < 10:
            return summary_dict, dict()

        # pid df, lines 8+
        pids_colnames = line_batch[8].split()
        pids_data = list(zip(*[line.split() for line in line_batch[9:]]))
        pids_dict = dict(zip(pids_colnames, pids_data))
        pids_dict["time"] = tuple([batch_time] * len(pids_data[0]))

        return summary_dict, pids_dict

    def plot_overview(self, *, between_time: tuple[str, str] | None = None) -> matplotlib.figure.Figure:
        """Create a summary figure visualizing top output."""
        assert self.summary_df.index.name == "time"  # relying on this index

        gridspec_kw = dict(top=0.92, right=0.8, hspace=0.08, height_ratios=[1, 2, 1, 2])
        fig, axs = plt.subplots(4, sharex=True, gridspec_kw=gridspec_kw, figsize=(12, 8))
        ycols_axs = dict(zip(["load_avg_1m", "%CPU", "avail_GiB", "%MEM"], axs))

        # plot
        my_summary = self.summary_df.between_time(*between_time) if between_time else self.summary_df
        for y, ax in ycols_axs.items():
            if y in ["avail_GiB", "load_avg_1m"]:
                self._plot_summary(my_summary, y=y, ax=ax)
            elif y == "%CPU":
                self._plot_pids(groupby="PID", y=y, ax=ax, between_time=between_time)
            elif y == "%MEM":
                self._plot_pids(groupby="job_name", y=y, ax=ax, between_time=between_time)

        time0 = my_summary.index[0]  # take date and timezone from first row
        self._format_axes(ycols_axs, xlabel=f"{self.summary_df.index.name} ({time0.tzname()})")
        fig.suptitle(f"{self.run_id} ({time0.date()})")
        plt.show(block=False)
        return fig

    def _plot_summary(self, my_summary, *, y, ax):
        color = "tab:gray"
        kwargs = dict(color=color, label="Total", linestyle="-", linewidth=1, marker="o", markersize=2)
        ax.plot(my_summary.index, my_summary[y], **kwargs)

        time0, ref0 = my_summary.iloc[0], self.summary_df.iloc[0].name
        self._plot_start_end(start=time0.name, ref_time=ref0, y=time0[y], ax=ax, color=color)

        timen, refn = my_summary.iloc[-1], self.summary_df.iloc[-1].name
        self._plot_start_end(end=timen.name, ref_time=refn, y=timen[y], ax=ax, color=color)

    def _plot_pids(self, *, groupby, y, ax, between_time=None):
        my_pid_stats = self.pids_df.between_time(*between_time) if between_time else self.pids_df
        groups = my_pid_stats.groupby(groupby)

        for pid, pid_name in self.pid_names.items():
            job_name = self.pid_jobs[pid]
            try:
                if groupby == "PID":
                    group_name, group = pid_name, groups.get_group(pid)
                elif groupby == "job_name":
                    group_name, group = job_name, groups.get_group(job_name).groupby(level=0)[[y]].sum()
            except KeyError:
                continue
            kwargs = dict(color=self.job_colors[job_name], zorder=self.pid_zorders[pid])
            self._plot_group(group_name, group, y=y, ax=ax, **kwargs)

            kwargs.update(zorder=self.pid_zorders[pid] + len(self.pid_zorders))
            time0, ref0 = group.iloc[0], self.pid_starts[pid]
            self._plot_start_end(start=time0.name, ref_time=ref0, y=time0[y], ax=ax, **kwargs)
            timen, refn = group.iloc[-1], self.pid_ends[pid]
            self._plot_start_end(end=timen.name, ref_time=refn, y=timen[y], ax=ax, **kwargs)

    @staticmethod
    def _plot_group(group_name, group, *, y, ax, **kwargs):
        linestyle = ":" if group_name.endswith("-worker") else "-"
        mykwargs = dict(
            alpha=0.75,
            label=group_name,
            marker="o",
            markersize=2,
            linestyle=linestyle,
            linewidth=1,
        )
        mykwargs.update(kwargs)
        ax.plot(group.index, group[y], **mykwargs)

    def _plot_start_end(self, *, y, ax, ref_time, start=None, end=None, **kwargs):
        if not (ref_time == start or ref_time == end):
            return
        mykwargs = dict(
            label="_",
            alpha=1,
            marker=">" if start else "s",
            markeredgewidth=2,
            markerfacecolor="w",
            markersize=8,
        )
        mykwargs.update(kwargs)
        ax.plot(start or end, y, **mykwargs)

    @staticmethod
    def _format_axes(ycols_axs, *, xlabel, bbox_to_anchor=(1.01, 0.96)):
        for y, ax in ycols_axs.items():
            handles, labels = ax.get_legend_handles_labels()
            legend_items = dict(zip(labels, handles))  # remove duplicate labels
            ax.legend(legend_items.values(), legend_items.keys(), loc="upper left", bbox_to_anchor=bbox_to_anchor)
            ax.set_ylabel(y)
            ax.tick_params(direction="inout")
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        ax.set_xlabel(xlabel)

    def plot_time_tags(self, summary_y: str = "used_GiB") -> matplotlib.figure.Figure:
        y_series = self.summary_df[summary_y]
        fig, ax = plt.subplots(figsize=(8, 6))

        kwargs = dict(s=100, zorder=5)
        for tag in self.time_tags.values():
            kwargs.update(color=tag.color, label=tag.name)
            ax.scatter(tag.time, y_series.loc[tag.time], **kwargs)

        kwargs = dict(color=self.default_color, label=f"Total {summary_y}", markersize=4)
        ax.plot(self.summary_df.index, y_series, "-o", **kwargs)

        xlabel = f"{self.summary_df.index.name} ({y_series.index[0].tzname()})"
        self._format_axes({summary_y: ax}, xlabel=xlabel, bbox_to_anchor=None)
        fig.suptitle(f"time_tags\n{self.run_id} ({y_series.index[0].date()})")
        plt.show(block=False)
        return fig


@attrs.define(kw_only=True)
class _TimeTag:
    name: str = attrs.field()
    time: pd.Timestamp = attrs.field()
    color: str | tuple[float, float, float] = attrs.field()
