import io
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='templates', static_folder='static')


def generate_sample_data():
    np.random.seed(42)
    n_samples = 100
    categories = ['A', 'B', 'C']

    data = {
        'category': np.random.choice(categories, n_samples),
        'feature1': np.random.randn(n_samples) * 10 + 50,
        'feature2': np.random.rand(n_samples) * 100,
        'feature3': np.random.randint(1, 100, n_samples),
        'feature4': np.random.randn(n_samples) * 5 + 25,
        'feature5': np.random.exponential(10, n_samples),
        'feature6': np.random.uniform(0, 1, n_samples)
    }

    df = pd.DataFrame(data)
    df['feature1'] = df['feature1'].round(2)
    df['feature2'] = df['feature2'].round(2)
    df['feature4'] = df['feature4'].round(2)
    df['feature5'] = df['feature5'].round(2)
    df['feature6'] = df['feature6'].round(4)

    return df


def create_parallel_coordinates(df, color_column=None):
    dimensions = []
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if color_column and color_column in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[color_column])
        if not is_numeric:
            unique_categories = df[color_column].unique().tolist()
            color_map = {cat: i for i, cat in enumerate(unique_categories)}
            color_values = df[color_column].map(color_map).tolist()
            showscale = True
            colorscale = 'Viridis'
        else:
            color_values = df[color_column].tolist()
            showscale = True
            colorscale = 'Viridis'
    else:
        color_values = [0] * len(df)
        showscale = False
        colorscale = 'Blues'

    for col in numeric_columns:
        dimensions.append({
            'label': col,
            'values': df[col].tolist(),
            'range': [df[col].min(), df[col].max()]
        })

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=color_values,
            colorscale=colorscale,
            showscale=showscale,
            cmin=min(color_values) if showscale else None,
            cmax=max(color_values) if showscale else None,
            colorbar=dict(
                title=color_column if color_column else ''
            ) if showscale else None
        ),
        dimensions=dimensions,
        labelfont=dict(size=12),
        tickfont=dict(size=10),
        rangefont=dict(size=10)
    ))

    fig.update_layout(
        title={
            'text': '平行坐标图 - Parallel Coordinates Plot',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18)
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=80, t=80, b=80),
        height=600
    )

    return json.loads(fig.to_json())


def dataframe_to_csv(df):
    return df.to_csv(index=False, encoding='utf-8')


def csv_to_dataframe(csv_content):
    df = pd.read_csv(io.StringIO(csv_content))
    return df


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    df = generate_sample_data()
    csv_data = dataframe_to_csv(df)
    return jsonify({
        'csv': csv_data,
        'columns': df.columns.tolist(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'row_count': len(df)
    })


@app.route('/api/generate-plot', methods=['POST'])
def generate_plot():
    try:
        data = request.get_json()
        csv_content = data.get('csv')
        color_column = data.get('color_column')

        if not csv_content:
            return jsonify({'error': 'No CSV data provided'}), 400

        df = csv_to_dataframe(csv_content)

        if len(df) == 0:
            return jsonify({'error': 'Empty dataset'}), 400

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) < 2:
            return jsonify({'error': 'At least 2 numeric columns required'}), 400

        plot_json = create_parallel_coordinates(df, color_column)

        return jsonify({
            'plot': plot_json,
            'columns': df.columns.tolist(),
            'numeric_columns': numeric_columns,
            'row_count': len(df),
            'summary': df.describe().to_json()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-columns', methods=['POST'])
def analyze_columns():
    try:
        data = request.get_json()
        csv_content = data.get('csv')

        if not csv_content:
            return jsonify({'error': 'No CSV data provided'}), 400

        df = csv_to_dataframe(csv_content)

        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()

            column_info.append({
                'name': col,
                'type': dtype,
                'is_numeric': is_numeric,
                'unique_count': int(unique_count),
                'null_count': int(null_count),
                'sample_values': df[col].dropna().head(3).tolist()
            })

        return jsonify({
            'columns': column_info,
            'row_count': len(df),
            'numeric_column_count': len(df.select_dtypes(include=[np.number]).columns)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
