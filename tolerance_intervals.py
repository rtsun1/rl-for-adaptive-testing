import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, wilcoxon, ttest_ind
from statsmodels.stats.multitest import multipletests

fig_1, ax = plt.subplots()

for seed in [111111, 222222]:
    for greedy in [False]:
        for time_aware in [True]:
            RMSE = []
            MAE = []
            bounds = [[], []]
            times = []
            for time in range(60, 60 + (3*60)*10, 60*3):
                times.append(time)
                data = np.load(f"Greedy_methods/{"greedy" if greedy else "ml"}_{"time_aware_" if time_aware else ""}{time}{"_"+str(seed) if seed != None else ""}.npy") #.npz
                vals = np.absolute(data[0] - data[1])
                MAE.append(vals.mean())
                RMSE.append(np.sqrt(np.mean(vals**2)))

                # Calculate the tolerance interval for the test losses
                # sorted_MAE = np.sort(vals).flatten()

                # confidence = 0.95
                # proportion = 0.90
                # v = len(vals) - binom.ppf(confidence, len(vals), proportion)
                # upper_index = np.ceil(len(vals) + 1 - v/2).astype(int)
                # lower_index = np.floor(v/2).astype(int)
                # bounds[0].append(sorted_MAE[lower_index])
                # bounds[1].append(sorted_MAE[upper_index])

            plt.figure(1)
            line = plt.plot(times, RMSE, label=f"Greedy {'EAP' if greedy else 'ML'} {'Time Aware' if time_aware else ''} {seed if seed != None else 'random'}")
            #ax.fill_between(times, bounds[0], bounds[1], color=line[0].get_color(), alpha=0.2)

            ax.relim()
            ax.autoscale_view()
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("RMSE")
            ax.set_title(f"Performance of Greedy Methods")# \n(Tolerance Interval: p={proportion}, 1-α={confidence})")
            handles, labels = plt.gca().get_legend_handles_labels()
            ax.legend()#handles[::-1], labels[::-1])
            ax.grid(True)
            plt.xticks(times, [str(int(t/60)) for t in times])
            
            fig_1.savefig("graphs/img/greedy_performance.png", dpi=300, bbox_inches='tight')

        
            RMSE = np.array(RMSE)
            np.save(f"ml_seed_{seed if seed != None else "random"}_time_aware_rmse", RMSE)
            for time, rmse in enumerate(RMSE):
                print(int(times[time]/60), "min: ", rmse)
            print(RMSE)
            print()

        # # compute wilcoxon test to mark the difference between the two models
        # stat, p = wilcoxon(TEST_LOSSES[0], TEST_LOSSES[1])
        # p_values.append(p)

        # reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        # plt.figure(2)
        # line3.set_data(num_questions, pvals_corrected)
        # line4.set_data(num_questions, [0.05] * len(num_questions))

        # ax_2.relim()
        # ax_2.autoscale_view()
        # ax_2.set_xlabel("Number of Random Questions")
        # ax_2.set_ylabel("p-value")
        # ax_2.set_title("Wilcoxon Signed-Rank Test p-values")
        # ax_2.legend()
        # ax_2.grid(True)
        # fig_2.savefig("graphs/img/wilcoxon_test.png", dpi=300, bbox_inches='tight')

        