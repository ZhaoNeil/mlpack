import plotly.express as px

# Read and parse the file
action_data = []
with open("src/mlpack/methods/reinforcement_learning/log/action_dist.txt", "r") as f:
    for line in f:
        if line.strip():
            actions = list(map(int, line.strip().split()))
            action_data.append(actions)

# Prepare data for Plotly
episodes = []
actions = []

for episode_idx, action_list in enumerate(action_data):
    for action in action_list:
        episodes.append(episode_idx)
        actions.append(action)

# Create the interactive scatter plot
fig = px.scatter(
    x=episodes,
    y=actions,
    labels={"x": "Episode", "y": "Action"},
    title="Interactive Action Distribution per Episode",
)

fig.update_layout(yaxis=dict(tickmode="linear", dtick=1))  # Ensure integer y-ticks

fig.show()
