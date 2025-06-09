import plotly.express as px

# Read and parse the file
state_data = []
with open("src/mlpack/methods/reinforcement_learning/log/state_space.txt", "r") as f:
    for line in f:
        if line.strip():
            states = list(map(float, line.strip().split()))
            state_data.append(states)

# Prepare data for Plotly
episodes = []
states = []

for episode_idx, state_list in enumerate(state_data):
    for state in state_list:
        episodes.append(episode_idx)
        states.append(state)

# Create the interactive scatter plot
fig = px.scatter(
    x=episodes,
    y=states,
    labels={"x": "Steps", "y": "State"},
    title="Interactive State Distribution per Episode",
)

fig.update_layout(yaxis=dict(tickmode="auto", dtick=1))  # Ensure integer y-ticks

fig.show()
