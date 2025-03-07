import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# 添加地理要素
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.gridlines()

# 设置中国区域
ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())

plt.show()