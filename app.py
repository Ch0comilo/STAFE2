import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.api import anova_lm
from scipy.stats import bartlett, shapiro
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

data = None

# Limpieza de imágenes antiguas
def limpiar_imagenes():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.endswith(".png"):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.remove(file_path)

# Ruta para subir el archivo CSV
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global data
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            limpiar_imagenes()
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            data = pd.read_csv(filepath)
            return redirect(url_for('select'))

    return render_template('upload.html')

# Ruta para seleccionar EDA o Modelo
@app.route('/select')
def select():
    return render_template('select.html')

# Ruta para el Análisis Exploratorio de Datos (EDA)
@app.route('/eda')
def eda():
    global data
    if data is None:
        flash("No se ha subido un archivo")
        return redirect(url_for('upload_file'))

    tables = data.head(10).to_html(classes='data')
    numerical_summary = data.describe().loc[['mean', 'std', 'min', '50%', 'max']].rename(index={'50%': 'median'}).to_dict()
    categorical_summary = {col: data[col].value_counts().to_dict() for col in data.select_dtypes(include=['object']).columns}
    charts = [col for col in data.columns if data[col].dtype != 'object']
    normality_scores = [(col, shapiro(data[col]).pvalue) for col in charts]

    # Crear gráficos
    for col in charts:
        plt.figure(figsize=(12, 6))
        ax1, ax2 = plt.subplot(121), plt.subplot(122)
        data[col].plot(kind='hist', bins=10, edgecolor='black', ax=ax1)
        ax1.set_title(f'Histograma de {col}')
        stats.probplot(data[col], dist="norm", plot=ax2)
        ax2.set_title(f'Q-Q de {col}')
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], f'{col}_plot.png'))
        plt.close()

    if len(charts) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(data[charts].corr(), annot=True, cmap='coolwarm', square=True)
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'correlation_heatmap.png'))
        sns.pairplot(data[charts], diag_kind='kde')
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'scatter_matrix.png'))
        plt.close()

    return render_template('EDA.html', tables=tables, numerical_summary=numerical_summary,
                           categorical_summary=categorical_summary, charts=charts,
                           normality_scores=sorted(normality_scores, key=lambda x: x[1], reverse=True),
                           heatmap_path='correlation_heatmap.png', pairplot_path='scatter_matrix.png')

# Ruta para el Modelo de Regresión
@app.route('/model')
def model():
    global data
    if data is None:
        flash("No se ha subido un archivo")
        return redirect(url_for('upload_file'))

    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    return render_template('MODEL.html', columns=numeric_columns, table=data.head().to_html(classes='table'))

@app.route('/regression', methods=['POST'])
def regression():
    global data
    if data is None:
        flash("No hay datos disponibles")
        return redirect(url_for('model'))

    response_var = request.form.get('response')
    predictor_vars = request.form.getlist('predictors')
    if response_var in predictor_vars:
        flash("La variable dependiente no puede estar entre las independientes")
        return redirect(url_for('model'))

    formula = f"{response_var} ~ {' + '.join(predictor_vars)}"
    model = smf.ols(formula=formula, data=data).fit()

    # Calcula el VIF solo si hay más de un predictor
    if len(predictor_vars) > 1:
        vif_data = {data[predictor_vars].columns[i]: variance_inflation_factor(data[predictor_vars].values, i)
                    for i in range(len(predictor_vars))}
        # Calcula el test de Bartlett solo si hay más de un predictor
        bartlett_stat, bartlett_p_value = bartlett(*[data[col] for col in predictor_vars])
    else:
        vif_data = "No es posible calcular el VIF con un solo predictor."
        bartlett_stat, bartlett_p_value = "No se puede calcular con un solo predictor", None

    residuals = model.resid
    fitted_values = model.fittedvalues
    shapiro_p_value = shapiro(residuals).pvalue
    durbin_watson_stat = sm.stats.durbin_watson(residuals)

    anova_table = anova_lm(model, typ=2).to_html()

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].hist(residuals, bins=30, edgecolor='k')
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[1, 0].boxplot(residuals, vert=False)
    axes[1, 1].scatter(fitted_values, residuals, alpha=0.7)
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'residual_plots.png'))
    plt.close()

    summary = {
        'response_var': response_var,
        'r_squared': model.rsquared,
        'residual_std_error': np.sqrt(model.mse_resid),
        'coefficients': model.params.to_dict(),
        'std_err': model.bse.to_dict(),
        'p_values': model.pvalues.to_dict(),
        'conf_intervals': model.conf_int().values.tolist(),
        'anova': anova_table,
        'shapiro_p_value': shapiro_p_value,
        'vif': vif_data,
        'bartlett_stat': bartlett_stat,
        'bartlett_p_value': bartlett_p_value,
        'durbin_watson_stat': durbin_watson_stat,
    }

    return render_template('MODEL.html', summary=summary)



if __name__ == "__main__":
    app.run(debug=True)