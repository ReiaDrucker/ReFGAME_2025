import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm

# Used to round ticks to 64
def round_to_nearest_value(data, val):
    return np.asarray([(e - e % val) if (e % val < (val / 2)) else (e + val - e % val) for e in data])

# Check whether a distribution is exponential or powerlaw
# Returns a tuple of the best fit and mle value calculated via negative log likelihood
def check_distribution_type(data):
    # General shape, location, scale family of distributions
    # alpha is the location parameter, which shifts the distribution along the x-axis.
    # Beta is the scale parameter, which determines the spread or scale of the distribution.
    # Gamma the shape parameter, which determines the shape of the distribution.
    # g is another function dependent on the standardized variable z and Beta
    # z = (x - a) / G
    # Fully f(x, g(z; B) ; a, B, G)

    # In scipy we generally we have (shape (optionally) , loc, scale)
    # For expon pars is (loc, scale) where g(x) = exp(-x)
    # For powerlaw we have (shape, loc, scale) where g(x,shape) = ax^(a-1) where a is the shape
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.powerlaw.html#scipy.stats.powerlaw
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit

    # The exponential distribution is a special case of the gamma distributions, with gamma shape = 1

    # warnings.filterwarnings("ignore")

    data = np.array(data)
    # Specify all distributions here
    # distributions = [st.powerlaw, st.expon]
    distributions = [st.powerlaw]
    errors = []
    parameters = []

    for distribution in distributions:
        pars = distribution.fit(data)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.nnlf.html#scipy.stats.rv_continuous.nnlf
        error = distribution.nnlf(pars, data)
        errors.append(error)
        parameters.append(pars)

    # Store all the data sorted by nnlf score (lower score is better)
    results = [(distribution.name, mle, p) for distribution, mle, p in
               sorted(zip(distributions, errors, parameters), key=lambda d: d[1])]

    best_fit = results[0]

    # warnings.filterwarnings("default")
    return best_fit[0], best_fit[1], best_fit[2], results

# Take in either the dwell or journey_dist_stats df and spit out the result of the dist fit
def compute_alpha_dist(df, is_rounds):
    results = []
    if is_rounds:
        for (matchID, side, player, mapName, roundNum, team), group in df.groupby(
                ['matchID', 'side', 'playerName', 'mapName', 'roundNum', 'team']):
            dist_type = check_distribution_type(group.duration)

            if dist_type[0] != 'powerlaw':
                results.append(
                    (matchID, side, player, mapName, team, roundNum, *dist_type[3][0][2], group.duration.shape[0],
                     dist_type[0], dist_type[3][0][1]))
            else:
                results.append(
                    (matchID, side, player, mapName, team, roundNum, *dist_type[3][0][2], group.duration.shape[0],
                     dist_type[0], dist_type[3][0][1]))

        return pd.DataFrame.from_records(results, columns=['matchID', 'side', 'playerName', 'mapName', 'team',
                                                           'roundNum',  'alpha', 'loc', 'scale',
                                                           'num_samples', 'best_fit_dist', 'MLE'])
    else:
        # Only look at the first half of a match
        df = df.loc[df['roundNum'] <= 15]

        for (matchID, side, player, mapName, team), group in df.groupby(
                ['matchID', 'side', 'playerName', 'mapName', 'team']):
            dist_type = check_distribution_type(group.duration)
            if dist_type[0] != 'powerlaw':
                results.append(
                    (matchID, side, player, mapName, team, *dist_type[3][0][2], group.duration.shape[0],
                     dist_type[0], dist_type[3][0][1]))
            else:
                results.append(
                    (matchID, side, player, mapName, team, *dist_type[3][0][2], group.duration.shape[0],
                     dist_type[0], dist_type[3][0][1]))

        return pd.DataFrame.from_records(results, columns=['matchID', 'side', 'playerName', 'mapName', 'team',
                                                           'alpha', 'loc', 'scale',
                                                           'num_samples', 'best_fit_dist', 'MLE'])

def create_feature_vectors(refgem_df, merged_alphas_df, refgem_features, alpha_features, is_rounds):

    if is_rounds:
        # Merge the two dataframes on common keys
        merge_keys = ['matchID', 'mapName', 'roundNum', 'playerName']
        merged_df = pd.merge(refgem_df, merged_alphas_df, on=merge_keys, how='inner')

        # Select only the needed features
        selected_features = refgem_features + alpha_features

        # Group by matchID, mapName, side, and roundNum
        grouped = merged_df.groupby(['matchID', 'mapName', 'side', 'roundNum'])

        feature_vectors = []
        labels = []
        feature_names = None  # Will initialize only once
        feature_vector_keys = []
        player_names_per_row = []

        for (matchID, mapName, side, roundNum), group in grouped:
            if len(group) != 5:
                continue  # Skip if not exactly 5 players

            feature_vector_keys.append((matchID, mapName, side, roundNum))

            # Sort players by name (to maintain consistent ordering)
            group_sorted = group.sort_values(by='playerName')

            # Extract player names
            player_names = group_sorted['playerName'].values
            player_names_per_row.append(player_names)

            # Extract features for each player and flatten
            player_features = group_sorted[selected_features].values.flatten()
            feature_vectors.append(player_features)
            labels.append((matchID, (mapName, side)))

            # Generate feature names once using p1, p2, ..., p5
            if feature_names is None:
                feature_names = []
                for i in range(5):  # p1 to p5
                    for feat in selected_features:
                        feature_names.append(f"p{i+1}_{feat}")

        return np.array(feature_vectors), labels, feature_names, feature_vector_keys, np.array(player_names_per_row)
    else:
        # Merge the two dataframes on common keys
        merge_keys = ['matchID', 'mapName', 'side', 'playerName']
        merged_df = pd.merge(refgem_df, merged_alphas_df, on=merge_keys, how='inner')

        # Select only the needed features
        selected_features = refgem_features + alpha_features

        # Group by matchID, mapName, side
        grouped = merged_df.groupby(['matchID', 'mapName', 'side', 'team'])

        feature_vectors = []
        labels = []
        feature_names = None  # Will initialize only once
        feature_vector_keys = []
        player_names_per_row = []

        for (matchID, mapName, side, team), group in grouped:
            if len(group) != 5:
                continue  # Skip if not exactly 5 players

            feature_vector_keys.append((matchID, mapName, side, team))

            # Sort players by name (to maintain consistent ordering)
            # Don't use individual player features for learning as is
            # You need an aggregation based feature or to use a deep-set architecture etc
            group_sorted = group.sort_values(by='playerName')

            # Extract player names
            player_names = group_sorted['playerName'].values
            player_names_per_row.append(player_names)

            # Extract features for each player and flatten
            player_features = group_sorted[selected_features].values.flatten()
            feature_vectors.append(player_features)
            labels.append(((matchID, team), (mapName, side)))

            # Generate feature names once using p1, p2, ..., p5
            if feature_names is None:
                feature_names = []
                for i in range(5):  # p1 to p5
                    for feat in selected_features:
                        feature_names.append(f"p{i + 1}_{feat}")

        return np.array(feature_vectors), labels, feature_names, feature_vector_keys, np.array(player_names_per_row)

if __name__ == '__main__':
    modes = ['camera-rounds', 'camera-halves', '3d-rounds', '3d-halves']

    # Restore string values using recovery map
    recovery_mapping = pd.read_parquet('../Outputs/recovery_mapping.parquet')
    recovered_mapping = (
        recovery_mapping.groupby('column_name')
        .apply(lambda g: dict(zip(g['code'], g['original_value'])))
        .to_dict()
    )

    for mode in tqdm(modes):
        mode_base, mode_detail = mode.split('-')

        if mode_base == 'camera':
            file_suffix = '-camera'
        else:
            file_suffix = ''

        d_df = pd.read_parquet(f'../Outputs/all_dwells{file_suffix}.parquet')
        j_df = pd.read_parquet(f'../Outputs/all_journeys{file_suffix}.parquet')

        d_metrics_df = pd.read_parquet(f'../Outputs/dwell_metrics{file_suffix}.parquet')
        j_metrics_df = pd.read_parquet(f'../Outputs/journey_metrics{file_suffix}.parquet')

        # Rename 'name' to 'playerName' to avoid ambiguity and reset index
        for df in [d_df, j_df, d_metrics_df, j_metrics_df]:
            df.rename(columns={'name': 'playerName'}, inplace=True)
            df.reset_index(drop=True, inplace=True)

        dwells_with_metrics_df = d_df.merge(d_metrics_df, on=['playerName', 'matchID', 'mapName', 'roundNum', f'dwellID{file_suffix}'], how='left')
        journeys_with_metrics_df = j_df.merge(j_metrics_df, on=['playerName', 'matchID', 'mapName', 'roundNum', f'journeyID{file_suffix}'], how='left')

        # Round to nearest half second (64 ticks)
        dwells_with_metrics_df.duration = round_to_nearest_value(dwells_with_metrics_df.duration, 64)
        journeys_with_metrics_df.duration = round_to_nearest_value(journeys_with_metrics_df.duration, 64)

        # Remove items less than 0.5 seconds (a single 2Hz tick) long
        dwells_with_metrics_df = dwells_with_metrics_df[dwells_with_metrics_df['duration'] > 0]
        journeys_with_metrics_df = journeys_with_metrics_df[journeys_with_metrics_df['duration'] > 0]

        # Not including fit quality metrics
        refgem_features = ['ch_area', 'ch_volume', 'ch_area_normed', 'ch_volume_normed', 'frac_dim', 'C1', 'C2', 'C3',
                           'C4', 'C5']
        alpha_features = ['alpha_d', 'num_samples_d', 'num_samples_j', 'alpha_j']
        refgem_df = pd.read_parquet(f'../Outputs/refgem_summary-{mode}.parquet')

        # Fix refgem_columns with the recovery map
        for col, mapping in recovered_mapping.items():
            if col in refgem_df.columns:
                refgem_df[col] = refgem_df[col].map(mapping)
        refgem_df = refgem_df.rename(columns={'name': 'playerName'})
        if 'side' in refgem_df.columns:
            refgem_df['side'] = refgem_df['side'].map({True: 'T', False: 'CT'})

        # Drop any columns that encountered an error during computation
        refgem_df = refgem_df.loc[refgem_df.has_issue == False]


        if mode_detail == 'rounds':
            dwell_dist_stats = compute_alpha_dist(dwells_with_metrics_df, is_rounds=True)
            journey_dist_stats = compute_alpha_dist(journeys_with_metrics_df, is_rounds=True)

            merged_alphas_df = pd.merge(
                dwell_dist_stats,
                journey_dist_stats,
                on=['mapName', 'playerName', 'side', 'roundNum', 'matchID'],
                suffixes=('_d', '_j')  # '_d' for dwell, '_j' for journey
            )

            # Variation without fit quality metrics
            feature_vectors, labels, feature_names, feature_keys, player_names_per_row = create_feature_vectors(
                refgem_df=refgem_df, merged_alphas_df=merged_alphas_df,
                refgem_features=refgem_features, alpha_features=alpha_features, is_rounds=True)

            # Save non-normalized feature vectors to a file
            # Treat inf as a large value since inf is not valid
            feature_vectors[np.isposinf(feature_vectors)] = 1e6

            processed_df = pd.DataFrame(feature_vectors, columns=feature_names)

            keys_df = pd.DataFrame(feature_keys, columns=['matchID', 'mapName', 'side', 'roundNum'])

            # Add player names as p1_name through p5_name
            player_names_df = pd.DataFrame(player_names_per_row, columns=[f'p{i + 1}_name' for i in range(5)])

            # Combine all parts
            processed_df = pd.concat([keys_df.reset_index(drop=True), player_names_df.reset_index(drop=True),
                                      processed_df.reset_index(drop=True)], axis=1)
            processed_df.loc[:, 'Label'] = processed_df['mapName'].astype(str) + '_' + processed_df['side'].astype(str)

            processed_df.to_parquet(f'../Outputs/cleaned-5-player-features-{mode}.parquet', index=True)

        else:
            dwell_dist_stats = compute_alpha_dist(dwells_with_metrics_df, is_rounds=False)
            journey_dist_stats = compute_alpha_dist(journeys_with_metrics_df, is_rounds=False)

            merged_alphas_df = pd.merge(
                dwell_dist_stats,
                journey_dist_stats,
                on=['mapName', 'playerName', 'side', 'matchID', 'team'],
                suffixes=('_d', '_j')  # '_d' for dwell, '_j' for journey
            )

            # Variation without fit quality metrics
            feature_vectors, labels, feature_names, feature_keys, player_names_per_row = create_feature_vectors(
                refgem_df=refgem_df, merged_alphas_df=merged_alphas_df,
                refgem_features=refgem_features, alpha_features=alpha_features, is_rounds=False)

            keys_df = pd.DataFrame(feature_keys, columns=['matchID', 'mapName', 'side', 'team'])

            # Save non-normalized feature vectors to a file
            # Treat inf as a large value since inf is not valid
            feature_vectors[np.isposinf(feature_vectors)] = 1e6
            processed_df = pd.DataFrame(feature_vectors, columns=feature_names)

            # Add player names as p1_name through p5_name
            player_names_df = pd.DataFrame(player_names_per_row, columns=[f'p{i + 1}_name' for i in range(5)])

            # Combine all parts
            processed_df = pd.concat([keys_df.reset_index(drop=True), player_names_df.reset_index(drop=True),
                                      processed_df.reset_index(drop=True)], axis=1)
            processed_df.loc[:, 'Label'] = processed_df['mapName'].astype(str) + '_' + processed_df['side'].astype(str)

            # Create a df indicating the starting sides in every game (technically should use the esta-lan file in case the first round of match has no journeys for any players on a team but that would be a cause for concern on its own so it should be fine)
            starting_sides_df = j_df[['matchID', 'mapName', 'side', 'team']].loc[j_df.roundNum == 1].groupby(
                ['matchID', 'mapName', 'side', 'team']).first().reset_index()

            # Only include the first halves of each game played
            processed_df = pd.merge(starting_sides_df, processed_df, on=['matchID', 'mapName', 'side', 'team'])

            processed_df.to_parquet(f'../Outputs/cleaned-5-player-features-{mode}.parquet', index=True)


