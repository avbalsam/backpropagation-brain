import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def get_csv_error_by_tau(tau: int):
    try:
        return pd.read_csv(f"random_walk/time_delay_tests_error_dependent/chessboard_error_time_delay:{tau}_numsquares:3_Net[2, 20, 20, 20, 1]/error_history.csv", index_col=0)
    except FileNotFoundError:
        return


def get_csv_error_by_tau_iteration(tau: int, iteration: int):
    try:
        return pd.read_csv(f"random_walk/time_delay_tests_error_dependent_iterations/chessboard_error_time_delay:{tau}_numsquares:3/iteration_{iteration}_Net[2, 20, 20, 20, 1]/error_history.csv", index_col=0)
    except FileNotFoundError:
        return


def get_error_by_tau_figure(tau_list):
    dfs = [(d := get_csv_error_by_tau(tau).rename(columns={"Error": f"{tau}"})).loc[:, d.columns != "Epoch"] for tau in tau_list]

    final_df = pd.concat(dfs, axis=1)

    plot = sns.lineplot(final_df, palette=sns.color_palette("rocket"))
    # plot.set_title("Convergence of Error by Time Delay")
    plot.set_xlabel("Batch (in thousands)")
    plot.set_ylabel("Error")

    lines = plt.gca().get_lines()

    # Change the linestyle of all lines to solid
    for line in lines:
        line.set_linestyle('-')

    # Change the legend's line styles to solid
    for legend_handle in plt.gca().get_legend().legendHandles:
        legend_handle.set_linestyle('-')

    plt.savefig("./figures/error_by_tau_figure.png")


def get_averaged_error_by_tau_figure(tau_list):
    dfs = []
    for tau in tau_list:
        iteration = 0
        ds = []
        while d := get_csv_error_by_tau_iteration(tau, iteration):
            print(f"Successfully found dataframe for tau={tau} and iteration={iteration}...")
            ds.append(d.rename(columns={"Error": f"{tau}"}).loc[:, d.columns != "Epoch"])
            iteration += 1

        dfs.append(pd.concat(ds).reset_index().groupby("index").mean())

    final_df = pd.concat(dfs, axis=1)

    plot = sns.lineplot(final_df, palette=sns.color_palette("rocket"))
    # plot.set_title("Convergence of Error by Time Delay")
    plot.set_xlabel("Batch (in thousands)")
    plot.set_ylabel("Error")

    lines = plt.gca().get_lines()

    # Change the linestyle of all lines to solid
    for line in lines:
        line.set_linestyle('-')

    # Change the legend's line styles to solid
    for legend_handle in plt.gca().get_legend().legendHandles:
        legend_handle.set_linestyle('-')

    plt.savefig("./figures/averaged_error_by_tau_figure.png")


def get_final_performance_by_tau_figure(max_tau):
    perf_list = []
    col_names = ["Time Delay", "Error After 50k Batches"]
    for tau in range(max_tau):
        try:
            df = get_csv_error_by_tau(tau)
            error_list = df["Error"].tolist()
            final_error = error_list[-1]
            perf_list.append([tau, final_error])
        except FileNotFoundError:
            pass

    final_df = pd.DataFrame(data=perf_list, columns=col_names)
    sns.lineplot(final_df, x="Time Delay", y="Error After 50k Batches")
    plt.savefig("./figures/final_performance_by_tau.png")


def get_averaged_final_performance_by_tau_figure(max_tau):
    # Not efficient at all, but I don't think that really matters
    perf_list = []
    col_names = ["Time Delay", "Error After 50k Batches"]
    for tau in range(max_tau):
        iteration = 0
        ds = []
        while d := get_csv_error_by_tau_iteration(tau, iteration):
            ds.append(d)

        if ds:
            df = pd.concat(ds).reset_index().groupby("index").mean()
            error_list = df["Error"].tolist()
            final_error = error_list[-1]
            perf_list.append([tau, final_error])

    final_df = pd.DataFrame(data=perf_list, columns=col_names)
    sns.lineplot(final_df, x="Time Delay", y="Error After 50k Batches")
    plt.savefig("./figures/final_performance_by_tau.png")


def generate_error_history_plot_from_csv(csv_filepath):
    """For plots without time delay tau"""
    df = pd.read_csv(csv_filepath, index_col=0)
    df = df.loc[:, df.columns != "Epoch"]
    plot = sns.lineplot(df)
    plot.set_xlabel("Batches (in thousands)")
    plt.savefig("./figures/error_history_chessboard_no_attention.png")



if __name__ == "__main__":
    get_error_by_tau_figure([0, 1, 5, 10, 20, 25])

    plt.close()

    get_final_performance_by_tau_figure(50)

    plt.close()

    generate_error_history_plot_from_csv("./random_walk/successful_test_without_attention/error_history.csv")

