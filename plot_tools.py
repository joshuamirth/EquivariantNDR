# Plot on Sphere
vis = X_ppca_iter[1].T
plot_title = 'PPCA Embedding of Klein Bottle'
# These settings give the "band" view.
azim = -171
elev = -15
x = np.cos(np.radians(azim))*np.sin(np.radians(90 - elev))
y = np.sin(np.radians(azim))*np.sin(np.radians(90 - elev))
z = np.cos(np.radians(90 - elev))
u = np.array([x,y,z])
u = u / np.linalg.norm(u)
vis = vis[sub_ind, :]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(plot_title)
ax.scatter(vis[:,0], vis[:,1], vis[:,2], c=c[sub_ind], cmap='plasma', alpha=0.5)
ax.scatter(-vis[:,0], -vis[:,1], -vis[:,2], c=c[sub_ind], cmap='plasma', alpha=0.5)
ax.quiver(0,0,0,u[0],u[1],u[2])
# ax.scatter(vis_sub[:,0], vis_sub[:,1], vis_sub[:,2], c='red', marker='+')
ax.view_init(elev, azim)
plt.show()

# 3D
vis = rotate_data(u, X_ppca_iter[1].T)
vis_sub = vis[sub_ind, :]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(plot_title)
ax.scatter(vis[:,0], vis[:,1], vis[:,2], c=cc, cmap='Greys')
ax.scatter(vis[:,0], vis[:,1], vis[:,2], c=c, cmap='plasma', alpha=0.3, linewidths=0)
# ax.scatter(vis_sub[:,0], vis_sub[:,1], vis_sub[:,2], c='red', marker='+')
ax.view_init(30,100)
plt.show()

# 2D
vis2 = stereographic(vis)
vis2_sub = vis2[sub_ind, :]
fig, ax = plt.subplots()
ax.scatter(vis2[:,0], vis2[:,1], c=c, cmap='Greys')
# ax.scatter(vis2[:,0], vis2[:,1], c=c, cmap='plasma', alpha=0.3, linewidths=0)
ax.scatter(vis2_sub[:,0], vis2_sub[:,1], c='red', marker='+')
draw_circle = plt.Circle((0,0), 1.0+buff/2, fill=False)
ax.add_artist(draw_circle)
ax.annotate("", xy=(1+buff/2, 0.04), xytext=(1+buff/2, 0.03), arrowprops=dict(arrowstyle="->"))
ax.annotate("", xy=(-1-buff/2, -0.02), xytext=(-1-buff/2, -0.01), arrowprops=dict(arrowstyle="->"))
ax.set_title(plot_title)
# ax.set(xlim=(-1-buff,1+buff), ylim=(-1-buff,1+buff))
ax.axis('equal')
fig.set_dpi(300)
# plt.savefig(pathname+'klein_bottle_points.png', dpi=300)
plt.show()