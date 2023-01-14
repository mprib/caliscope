#%%

import pandas as pd
from pathlib import Path


repo = str(Path(__file__)).split("src")[0]
output_directory = Path(repo, "sessions", "iterative_adjustment")

pre_file = Path(output_directory, "recording", "triangulated_points_daisy_chain.csv")
post_file = Path(
    output_directory, "recording", "triangulated_points_bundle_adjusted.csv"
)


post_points = pd.read_csv(post_file)

#%%

def std_hist(file_path):
    points = pd.read_csv(file_path).rename( columns={"x_pos": "x", "y_pos": "y", "z_pos": "z"})
    std_points = (points[["bundle", "id", "pair", "x", "y", "z"]]
                    .groupby(["bundle", "id"])
                    .agg({"x":"std", "y":"std", "z":"std", "pair":"size"})
                    .rename(columns={"x":"x_std", "y":"y_std", "z":"z_std", "pair":"count"})

                    # .reset_index()
                    )

    # limit to observations with all three cameras seeing points
    std_points = std_points[std_points["count"]>1].drop(columns=["count"])

    std_points.hist(bins=100)


# %%

def euclidean_dist(file_path):
    points = pd.read_csv(file_path).rename( columns={"x_pos": "x", "y_pos": "y", "z_pos": "z"})
    mean_points = (points[["bundle", "id", "pair", "x", "y", "z"]]
                    .groupby(["bundle", "id"])
                    .agg({"x":"mean", "y":"mean", "z":"mean", "pair":"size"})
                    .rename(columns={"x":"x_mean", "y":"y_mean", "z":"z_mean", "pair":"count"})
                    .reset_index()
                    )
    # limit to observations with all three cameras seeing points
    mean_points = mean_points[mean_points["count"]>1].drop(columns=["count"])

    merged_points = (points
                    .merge(mean_points, how="left", on=["bundle","id"])
                    .dropna())


    def euclidian_distance(row):
        x_dist = (row["x"] - row["x_mean"])
        y_dist = (row["y"] - row["y_mean"])
        z_dist = (row["z"] - row["z_mean"])
   
        return (x_dist**2 + y_dist**2 + z_dist**2)**(0.5) 
   
    
    merged_points = merged_points[["x", "y","z", "x_mean", "y_mean", "z_mean"]]

    merged_points["euclidian_distance"] = merged_points.apply(euclidian_distance, axis = 1)
    merged_points["euclidian_distance_cm"] = merged_points["euclidian_distance"] *100

    return merged_points["euclidian_distance_cm"]
    

pre_distance = euclidean_dist(pre_file)
post_distance = euclidean_dist(post_file)
# %%
pre_distance.hist(bins=100)

#%%
post_distance.hist(bins=100)

#%%
pre_distance = pre_distance.to_frame()
post_distance = post_distance.to_frame()
pre_distance["measure_type"] = ["PreBundleAdjustment"]*len(pre_distance)
post_distance["measure_type"] = ["PostBundleAdjustment"]*len(post_distance)

combined_distances = pd.concat([pre_distance,post_distance])

# %%
combined_distances.plot.box(by="measure_type")
pre_mean = pre_distance.mean(numeric_only=True).values[0]
post_mean = post_distance.mean(numeric_only=True).values[0]
print(f"PRE BUNDLE ADJUSTMENT: Mean distance to centroid of {pre_mean} cm")
print(f"POST BUNDLE ADJUSTMENT: Mean distance to centroid of {post_mean} cm")

# %%
