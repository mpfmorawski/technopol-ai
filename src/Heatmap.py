import matplotlib.image as mpimg
import numpy as np
import plotly.express as px
from PIL import Image
import base64

class Heatmap():
    """
    Class for data visualization with a heatmap
    """

    def __init__(self, df, col_name):
        self.df = df
        self.col_name = col_name
        self.lat = self.df["Latitude"].to_numpy()
        self.lng = self.df["Longitude"].to_numpy()
        self.min_lat = min(self.lat)
        self.max_lat = max(self.lat)
        self.min_lng = min(self.lng)
        self.max_lng = max(self.lng)

    def get_2d_plot(self):
        img = Image.open('./data/nyc_big.png')
        img.load()

        with open('./data/nyc_big_left.png', "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        encoded_image = "data:image/png;base64," + encoded_string

        # values = self.df[self.col_name].to_numpy()
        fig = px.scatter(self.df, x='Longitude', y='Latitude', color=self.col_name, hover_data={self.col_name: True})
        fig.add_layout_image(
        dict(
            source=encoded_image,
            xref="x",
            yref="y",
            x=self.min_lng,
            y=self.max_lat,
            sizex=self.max_lng-self.min_lng,
            sizey=self.max_lat-self.min_lat,
            sizing="stretch",
            opacity=0.8,
            layer="below")
        )
        width = self.max_lng - self.min_lng
        height = self.max_lat - self.min_lat
        fig.update_layout(
            autosize = False,
            height = 800,
            width = 800 * width / height,
            scene = dict(aspectmode='cube'),
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis =  { 'showgrid': False },
            yaxis = { 'showgrid': False }
        )
        # fig_bytes = fig.to_image(format="png")
        # buf = io.BytesIO(fig_bytes)
        # img = Image.open(buf)
        # return np.asarray(img)
        return fig

    def get_3d_plot(self):
        img = mpimg.imread('./data/nyc_big.png')[:,:,1]
        y, x = np.meshgrid(np.linspace(self.min_lat, self.max_lat, img.shape[1]), np.linspace(self.min_lng, self.max_lng, img.shape[0]))
        z = np.full(x.shape, 0.01)

        values = self.df[self.col_name].to_numpy()
        colors = [np.sqrt(val / max(values)) for val in values]
        fig = px.scatter_3d(x=self.lng, y=self.lat, z=values, color=colors)
        fig.add_surface(x=x, y=y, z=z, 
                surfacecolor=img, 
                colorscale='earth', 
                showscale=False)
        fig.update_layout(
            scene = dict(aspectratio=dict(x=10, y=10, z=1)),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        return fig