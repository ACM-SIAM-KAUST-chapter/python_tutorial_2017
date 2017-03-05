# Visualize the data
x_index = 7
y_index = 12

plt.figure(figsize=(7, 5))
plt.scatter(data.data[:, x_index], data.data[:, y_index], c=data.target)
plt.colorbar(ticks=range(51))
plt.xlabel(data.feature_names[x_index])
plt.ylabel(data.feature_names[y_index])
