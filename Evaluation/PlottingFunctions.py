from functools import reduce
from itertools import zip_longest
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer


def __ax_pretty(
        sns_ax,
        target_path,
        title_string="",
        sub_title_string="",
        show=True,
        special_x_label=None,
        special_y_label=None
):
    plt.suptitle(title_string)
    plt.title(sub_title_string)
    if special_x_label is not None:
        sns_ax.set_xlabels(special_x_label.title())
    else:
        sns_ax.set_xlabels(sns_ax.ax.get_xlabel().replace("_", " ").title())
    if special_y_label is not None:
        sns_ax.set_ylabels(special_y_label.title())
    else:
        sns_ax.set_ylabels(sns_ax.ax.get_ylabel().replace("_", " ").title())
    sns_ax.tight_layout()
    plt.savefig(target_path)
    if show:
        plt.show()
    else:
        plt.close()


def _make_plot(
        data: pd.DataFrame,
        x_var: str,
        y_var: str,
        title_string: str,
        sub_title_string: str,
        special_x_label: str,
        special_y_label: str,
        plot_folder: Path,
        df_names: Optional[List[str]] = None,
        show: bool = True
) -> None:
    if df_names is not None:
        renaming_map = {f"{y_var}_TABLE{i}": df_names[i] for i in range(len(df_names))}
    else:
        renaming_map = {}

    data = (
        pd.concat(
            [
                (data
                 .filter(regex=f"^{col_name}$|^{y_var}$|^{y_var}_TABLE\d*$")
                 .rename(columns=renaming_map)
                 .melt([col_name])
                 .rename(columns={col_name: x_var})
                 )
                for col_name in data.filter(like=x_var).columns
            ]
        )
    )

    data_without_y = ((data
                       .where(lambda row: row["variable"] != y_var, axis="rows"))
                      .dropna()
                      )

    if not data_without_y.empty:
        data = data_without_y

    ax = sns.relplot(data=data, x=x_var, y="value", hue="variable", s=10, facet_kws=dict(legend_out=False))

    if len(ax.legend.get_texts()) > 1:
        # build legend: 
        # calculate font size in inches and estimate characterwidth with 0.65 * height
        factor = ax.legend._fontsize * 1 / 72 * 0.7
        max_label_length = max(max(map(len, df_names)) * factor, 7)
        # calculate aspect between height and width
        inch_sizes_of_fig = ax.figure.get_size_inches()
        aspect = inch_sizes_of_fig[1] / inch_sizes_of_fig[0]
        # set figure to fit legend
        ax.figure.set_size_inches(max_label_length, aspect * max_label_length)
        ax.ax.legend(
            loc="upper left",
            title=None,
            bbox_to_anchor=(
                0,
                -ax.legend._fontsize * 1 / 72 * 5 / ax.figure.get_size_inches()[1]),
            handletextpad=0.2
        )
        plt.tight_layout()
    else:
        ax.legend.remove()

    __ax_pretty(
        sns_ax=ax,
        target_path=(plot_folder / title_string),
        title_string=title_string,
        sub_title_string=sub_title_string,
        special_x_label=special_x_label,
        special_y_label=special_y_label,
        show=show
    )


def _all_plots(
        data: pd.DataFrame,
        plot_folder: Path,
        sub_title_string: str,
        data_names: Optional[list[str]],
        emission_plotting: bool,
        winning_layer_plotting: bool,
        show_only_in_development: bool = True
) -> None:
    in_development = True
    not_in_development = not show_only_in_development

    table_names = (data.filter(like='shape_of_hidden_layers').columns.map(
        lambda s: s.replace('shape_of_hidden_layers', '')).values.tolist())

    title_string = "Instances trained on vs Accuracy".title()

    _make_plot(
        data=data,
        x_var="instances",
        y_var="accuracy",
        title_string=title_string,
        sub_title_string=sub_title_string,
        special_x_label="instances",
        special_y_label="accuracy",
        plot_folder=plot_folder,
        df_names=data_names,
        show=not_in_development
    )

    title_string = "Amount of active output layers vs Accuracy".title()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_seq_items", None)

    # add the number of active layers to the dataframe
    active_layers = data.filter(like="active_layers")
    new_col_names_amounts_of_active_layer = [f"amount_of_active_layers{col_name.replace('active_layers', '')}" for
                                             col_name in
                                             active_layers.columns]
    data[new_col_names_amounts_of_active_layer] = active_layers.map(len)

    _make_plot(
        data=data,
        x_var="amount_of_active_layers",
        y_var="accuracy",
        title_string=title_string,
        sub_title_string=sub_title_string,
        special_y_label="Accuracy",
        special_x_label="Amount of Active Layers",
        plot_folder=plot_folder,
        df_names=data_names,
        show=not_in_development
    )

    title_string = "Amount of Hidden Layers vs Accuracy".title()
    _make_plot(
        data=data,
        x_var="nr_of_layers",
        y_var="accuracy",
        title_string=title_string,
        sub_title_string=sub_title_string,
        special_x_label="Amount of Hidden Layers",
        special_y_label="Accuracy",
        plot_folder=plot_folder,
        df_names=data_names,
        show=not_in_development
    )

    title_string = "Amount of Hidden Nodes vs Accuracy".title()
    amount_of_hidden_nodes = (data
                              .filter(like="shape_of_hidden_layers")
                              .map(lambda list_of_shapes: sum((layer_shape[1] for layer_shape in list_of_shapes)))
                              )
    new_col_names_amounts_of_active_layer = [
        f"amount_of_hidden_nodes{col_name.replace('shape_of_hidden_layers', '')}"
        for col_name in amount_of_hidden_nodes.columns
    ]
    data[new_col_names_amounts_of_active_layer] = amount_of_hidden_nodes
    _make_plot(
        data=data,
        x_var="amount_of_hidden_nodes",
        y_var="accuracy",
        title_string=title_string,
        sub_title_string=sub_title_string,
        special_x_label="Amount of Nodes in Hidden Layers",
        special_y_label="Accuracy",
        plot_folder=plot_folder,
        df_names=data_names,
        show=not_in_development
    )

    title_string = "Instances vs Amount of Hidden Layers".title()
    _make_plot(
        data=data,
        x_var="instances",
        y_var="nr_of_layers",
        title_string=title_string,
        sub_title_string=sub_title_string,
        special_x_label="Instances",
        special_y_label="Amount of Hidden Layers",
        plot_folder=plot_folder,
        df_names=data_names,
        show=not_in_development
    )

    title_string = "Instances vs Amount of Hidden Nodes".title()
    _make_plot(
        data=data,
        x_var="instances",
        y_var="amount_of_hidden_nodes",
        title_string=title_string,
        sub_title_string=sub_title_string,
        special_x_label="Instances",
        special_y_label="Amount of Nodes in Hidden Layers",
        plot_folder=plot_folder,
        df_names=data_names,
        show=not_in_development
    )

    title_string = "Instances vs Amount of Active Layers"
    _make_plot(
        data=data,
        x_var="instances",
        y_var="amount_of_active_layers",
        title_string=title_string,
        sub_title_string=sub_title_string,
        special_x_label="Instances",
        special_y_label="Amount of Active Layers",
        plot_folder=plot_folder,
        df_names=data_names,
        show=not_in_development
    )

    title_string = "Instances vs Amount of Nodes of Active Layers"
    amount_of_active_hidden_nodes = (data
    .filter(regex="^shape_of_hidden_layers$|^shape_of_hidden_layers_TABLE\d+$|^active_layers$|^active_layers_TABLE\d+$")
    .apply(
        func=lambda row: [
            ((np.array(row.loc[f"shape_of_hidden_layers{col_name}"])
              .take(row.loc[f"active_layers{col_name}"], axis=0)
              .take(True, axis=1)
              )
             .sum())
            for col_name in table_names
        ],
        axis="columns",
        result_type="expand"
    )
    )
    new_col_names_amounts_of_active_hidden_nodes = [
        f"amount_of_active_hidden_nodes{col_name}"
        for col_name in table_names
    ]
    data[new_col_names_amounts_of_active_hidden_nodes] = amount_of_active_hidden_nodes
    _make_plot(
        data=data,
        x_var="instances",
        y_var="amount_of_active_hidden_nodes",
        title_string=title_string,
        sub_title_string=sub_title_string,
        special_x_label="Instances",
        special_y_label="Amount of Active Hidden Nodes",
        plot_folder=plot_folder,
        df_names=data_names,
        show=not_in_development
    )

    title_string = "Which layer is active vs instance that is trained".title()
    data_with_exploded_active_layers = (data
                                        .loc[:, "instances"]
                                        .to_frame()
                                        .join(data
                                              .apply(
        lambda row: list(zip_longest(*[row[col_name] for col_name in active_layers.columns])), axis=1)
                                              .explode()
                                              .apply(lambda row: pd.Series(row, index=active_layers.columns))
                                              .groupby(level=0)
                                              .ffill()
                                              )
                                        )

    _make_plot(
        data=data_with_exploded_active_layers,
        x_var="instances",
        y_var="active_layers",
        title_string=title_string,
        sub_title_string=sub_title_string,
        special_x_label="Instances",
        special_y_label="identity of active layers".title(),
        plot_folder=plot_folder,
        df_names=data_names,
        show=not_in_development
    )

    title_string = "lifetime of output layers in amount of instances".title()
    counts = (data_with_exploded_active_layers
              .melt(["instances"])
              .value_counts(["value", "variable"])
              .to_frame()
              .reset_index()
              .pivot(index="value", columns="variable")
              .reset_index()
              )
    counts.columns = counts.columns.get_level_values(1)
    counts.rename(columns={"": "layer_id"}, inplace=True)

    _make_plot(
        data=counts,
        x_var="layer_id",
        y_var="active_layers",
        title_string=title_string,
        sub_title_string=sub_title_string,
        special_x_label="identity of active layers".title(),
        special_y_label="Lifetime [Amount of Instances]".title(),
        plot_folder=plot_folder,
        df_names=data_names,
        show=not_in_development
    )

    if emission_plotting:
        title_string = "Instances vs Emissions".title()
        _make_plot(
            data=data,
            x_var="instances",
            y_var="emissions",
            title_string=title_string,
            sub_title_string=sub_title_string,
            special_x_label="Instances",
            special_y_label="Emissions",
            plot_folder=plot_folder,
            df_names=data_names,
            show=not_in_development
        )

        title_string = "Amount of Active Layers vs Emissions".title()
        _make_plot(
            data=data,
            x_var="amount_of_active_layers",
            y_var="emissions",
            title_string=title_string,
            sub_title_string=sub_title_string,
            special_x_label="Amount of Active Layers",
            special_y_label="Emissions",
            plot_folder=plot_folder,
            df_names=data_names,
            show=not_in_development
        )

        title_string = "Amount of Hidden Nodes vs emissions".title()
        _make_plot(
            data=data,
            x_var="amount_of_hidden_nodes",
            y_var="emissions",
            title_string=title_string,
            sub_title_string=sub_title_string,
            special_x_label="Amount of Nodes in Hidden Layers",
            special_y_label="Emissions",
            plot_folder=plot_folder,
            df_names=data_names,
            show=not_in_development
        )

    if winning_layer_plotting:
        title_string = "Trained Layer on Instance".title()

        _make_plot(
            data=data,
            x_var="instances",
            y_var="winning_layer",
            special_x_label="Instances".title(),
            special_y_label="Identity of Winning Layer".title(),
            title_string=title_string,
            sub_title_string=sub_title_string,
            df_names=data_names,
            plot_folder=plot_folder,
            show=not_in_development
        )
        title_string="Trained Layer on Instance for layers active at the end".title()
        surviving_layers = data.filter(regex="^active_layers(_TABLE\d+)*$").tail(1).reset_index(drop=True)

        winning_layer_filtered_for_surviving_layers = (
            pd.concat(
                [
                    (
                        data
                        .filter(regex=f"^winning_layer{tbl_name}$")
                        .where(lambda row: row[f"winning_layer{tbl_name}"].isin(surviving_layers.loc[0, f"active_layers{tbl_name}"]), axis="rows")
                        .dropna()
                    )
                    for tbl_name in table_names
                ]
            )
            .reset_index()
        )

        _make_plot(
            data=winning_layer_filtered_for_surviving_layers,
            x_var="index",
            y_var="winning_layer",
            special_x_label="Instances".title(),
            special_y_label="Identity of Winning Layer".title(),
            title_string=title_string,
            sub_title_string=sub_title_string,
            df_names=data_names,
            plot_folder=plot_folder,
            show=not_in_development
        )

        title_string = "Amount of Winning Layer Trainings on Layer".title()
        amount_of_trainings_per_layer = (
            pd.concat(
                [
                    (data
                     .filter(regex=f"^winning_layer{tbl_name}$")
                     .value_counts()
                     .reset_index()
                     .melt("count")
                     .pivot(index="value", columns="variable")
                     .reset_index()
                     )
                    for tbl_name in table_names
                ]
            )
        ).fillna(0)
        amount_of_trainings_per_layer.columns = amount_of_trainings_per_layer.columns.get_level_values(1)
        amount_of_trainings_per_layer.rename(columns={"": "layer_id"}, inplace=True)
        _make_plot(
            data=amount_of_trainings_per_layer,
            x_var="layer_id",
            y_var="winning_layer",
            special_x_label="identity of active layers".title(),
            special_y_label="amount of trainings layer".title(),
            title_string=title_string,
            sub_title_string=sub_title_string,
            plot_folder=plot_folder,
            df_names=data_names,
            show=not_in_development
        )


def __plot_and_save_result(result_id: int, show: bool = True) -> None:
    results_dir_path = Path(f"results/runs/runID={result_id}")
    if not results_dir_path.exists():
        print(f"runID={result_id}: No results found, returning")
        return
    hyperparameter_folders = list(results_dir_path.iterdir())
    if not any(hyperparameter_folders):
        print(f"runID={result_id}: No test run with hyperparameters found, returning")
        return

    for hyperparameter_folder in hyperparameter_folders:
        datastream_folders = list(hyperparameter_folder.iterdir())
        if not any(datastream_folders):
            print(
                f"runID={result_id}: No test run with a datastream found for hyperparameter={hyperparameter_folder.name}, skipping")
            continue

        for datastream_folder in datastream_folders:
            metrics_overview_path = datastream_folder / "metrics.pickle"

            if not metrics_overview_path.exists():
                print(
                    f"runID={result_id}: No metrics overview file found for hyperparameter={hyperparameter_folder.name} and datastream={datastream_folder.name}, skipping")
                continue

            metrics_overview = pd.read_pickle(metrics_overview_path)

            all_metrics_path = datastream_folder / "metrics_per_window.pickle"
            if not all_metrics_path.exists():
                print(
                    f"runID={result_id}: No metrics file found for hyperparameter={hyperparameter_folder.name} and datastream={datastream_folder.name}, skipping")
                continue

            results_csv = pd.read_pickle(all_metrics_path)
            emissions_path = datastream_folder / "emissions.csv"
            emissions_plotting = False
            if emissions_path.exists():
                emissions = pd.read_csv(emissions_path)
                results_csv = results_csv.merge(emissions, right_index=True, left_index=True)
                emissions_plotting = True

            # plotting
            lr, mci_cut, *_ = map(lambda s: s.split("=")[1], hyperparameter_folder.name.split("_"))
            lr, mci_cut = float(lr), float(mci_cut)
            sub_title_string = f"instances={metrics_overview.instances[0]:n}, lr={lr:.2e}, mci-cut-off={mci_cut :.2e},\nmean-accuracy={metrics_overview.accuracy[0]:.2f},\nnr of active layers after training={metrics_overview.active_layers.map(len)[0]:n}".title()
            plot_folder = datastream_folder / "plots"
            plot_folder.mkdir(exist_ok=True)

            contains_winning_layer_column = len(results_csv.filter(regex="^winning_layer$").columns) > 0

            _all_plots(
                data=results_csv,
                plot_folder=plot_folder,
                sub_title_string=sub_title_string,
                emission_plotting=emissions_plotting,
                data_names=None,
                show_only_in_development=(not show),
                winning_layer_plotting=contains_winning_layer_column
            )


def __create_df_names(df_parameters: List[Dict[str, str]]) -> Tuple[str, List[str]]:
    common_keys = set(
        filter(lambda k: len(set(map(lambda df_dict: df_dict[k], df_parameters))) == 1, df_parameters[0].keys()))
    substring = "\n ".join((f"{key}={df_parameters[0][key]}" for key in common_keys))
    return substring, [
        ", ".join((f"{key}={value}" for key, value in df_parameter_dict.items() if key not in common_keys)) for
        df_parameter_dict in df_parameters]


def __get_plot_folder(sup_title, df_names) -> Path:
    comp_description = (sup_title + "\ncompare: [(" + "), (".join(df_names) + ")]")
    working_folder = Path("results") / "comparisons"
    working_folder.mkdir(exist_ok=True, parents=True)
    if len(list(working_folder.iterdir())) == 0:
        plot_folder = working_folder / "comparison=0" / "plots"
    else:
        plot_folder = working_folder / f"comparison={len(list(working_folder.iterdir()))}" / "plots"
        i = 0
        while plot_folder.parent.exists():
            i += 1
            plot_folder = working_folder / f"comparison={len(list(working_folder.iterdir())) + i}" / "plots"
    plot_folder.mkdir(exist_ok=True, parents=True)

    with (plot_folder.parent.resolve() / "description").open("w") as f:
        f.write(comp_description)

    return plot_folder


def __compare_results_via_plot_and_save(result_paths: List[Path], show: bool = True) -> None:
    dfs = []
    df_names = []
    df_contains_emission_flags = []
    df_contains_winning = []

    for i, result_dir in enumerate(result_paths):
        metrics_windowed_path, _, *tail = [file for file in result_dir.iterdir() if file.is_file()].__reversed__()
        run_data = pd.read_pickle(metrics_windowed_path)
        contains_emissions = False
        if tail:
            emissions = pd.read_csv(tail[0])
            run_data = run_data.merge(emissions, right_index=True, left_index=True)
            contains_emissions = True

        run_data.name = f"TABLE{i}"
        df_names.append(
            f"{result_dir.parent.parent.name}, {result_dir.parent.name.replace('_', ', ')}, stream={result_dir.name}")

        contains_winning_layer_column = len(run_data.filter(regex="^winning_layer$").columns) > 0

        dfs.append(run_data)
        df_contains_winning.append(contains_winning_layer_column)
        df_contains_emission_flags.append(contains_emissions)

    def merge_two_dfs(lhs, rhs):
        out = pd.merge(lhs, rhs, on="instances", how="outer", suffixes=("_" + lhs.name, "_" + rhs.name))
        out.name = rhs.name
        return out

    data = reduce(merge_two_dfs, dfs)
    df_parameters = [dict((tuple(part.split("=")) for part in name.split(", "))) for name in df_names]
    sub_title_string, shortened_df_names = __create_df_names(df_parameters)

    plot_folder = __get_plot_folder(sub_title_string, shortened_df_names)

    _all_plots(
        data=data,
        plot_folder=plot_folder,
        sub_title_string=sub_title_string,
        emission_plotting=all(df_contains_emission_flags),
        data_names=shortened_df_names,
        show_only_in_development=not show,
        winning_layer_plotting=all(df_contains_winning)
    )


def __compare_all_of_one_run(run_id: int, show: bool) -> None:
    all_results = []
    results_dir_path = Path(f"results/runs/runID={run_id}")
    if not results_dir_path.exists():
        print(f"runID={run_id}: No results found, returning")
        return
    hyperparameter_folders = list(results_dir_path.iterdir())
    if not any(hyperparameter_folders):
        print(f"runID={run_id}: No test run with hyperparameters found, returning")
        return

    for hyperparameter_folder in hyperparameter_folders:
        datastream_folders = list(hyperparameter_folder.iterdir())
        if not any(datastream_folders):
            print(
                f"runID={run_id}: No test run with a datastream found for hyperparameter={hyperparameter_folder.name}, skipping")
            continue

        for datastream_folder in datastream_folders:
            metrics_overview_path = datastream_folder / "metrics.pickle"

            if not metrics_overview_path.exists():
                print(
                    f"runID={run_id}: No metrics overview file found for hyperparameter={hyperparameter_folder.name} and datastream={datastream_folder.name}, skipping")
                continue

            all_metrics_path = datastream_folder / "metrics_per_window.pickle"
            if not all_metrics_path.exists():
                print(
                    f"runID={run_id}: No metrics file found for hyperparameter={hyperparameter_folder.name} and datastream={datastream_folder.name}, skipping")
                continue
            all_results.append(datastream_folder)

    __compare_results_via_plot_and_save(all_results, show)
