from collections import Counter
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import dash_table
from pipeline import pipeline


final_predict, clusters_names = pipeline()
count = Counter(final_predict)
top_words = pd.read_csv("../dataset/top10words.csv")

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

Title = html.H1(children="News Clusters", style={"textAlign": "center",})
Bar = dcc.Graph(
    id="clussters",
    figure={
        "data": [
            {
                "x": [clusters_names[key] for key in count.keys()],
                "y": list(count.values()),
                "type": "bar",
            }
        ]
    },
)


Table_title = html.Div(
    children="Top 10 keywords in clusters", style={"textAlign": "center",}
)
Table = dash_table.DataTable(
    id="top_words",
    columns=[{"name": i, "id": i} for i in top_words.columns],
    data=top_words.to_dict("records"),
)


app.layout = html.Div(children=[Title, Bar, Table_title, Table])
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080, debug=True)
