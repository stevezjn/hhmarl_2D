import cartopy.crs as ccrs
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
plt.show()