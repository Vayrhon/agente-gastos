from flask import Flask, render_template, request, redirect
import sqlite3
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


app = Flask(__name__)
DB = "gastos.db"

# Crear tablas si no existen
conn = sqlite3.connect(DB)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS gastos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT,
    categoria TEXT,
    monto REAL,
    comentario TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS presupuestos (
    categoria TEXT PRIMARY KEY,
    monto_max REAL
)
""")
conn.commit()
conn.close()

@app.route('/')

def index():
    conn = sqlite3.connect(DB)
    
    # Últimos 10 movimientos
    df = pd.read_sql_query("SELECT * FROM gastos ORDER BY fecha DESC LIMIT 10", conn)
    
    # Todos los movimientos (para el gráfico)
    df_todos = pd.read_sql_query("SELECT * FROM gastos", conn)

    pres_df = pd.read_sql_query("SELECT * FROM presupuestos", conn)
    conn.close()

    total = df['monto'].sum() if not df.empty else 0

    # Crear dict de presupuestos
    pres_dict = dict(zip(pres_df['categoria'], pres_df['monto_max']))

    # Calcular gastos totales por categoría (usando TODOS los movimientos)
    gastos_cat = df_todos.groupby('categoria')['monto'].sum().to_dict()
    total_presupuesto = sum(pres_dict.values())
    # Preparar data para el gráfico de torta
    grafico_data = []
    for cat, presupuesto in pres_dict.items():
        if presupuesto > 0:
            gastado = gastos_cat.get(cat, 0)
            disponible = presupuesto - gastado
            disponible = max(disponible, 0)  # evitar negativos

            grafico_data.append({
                'categoria': cat,
                'gastado': round(gastado, 2),
                'disponible': round(disponible, 2)
            })


    no_presupuestos = len(pres_dict) == 0

    return render_template("index.html",
                        gastos=df.to_dict(orient="records"),
                        total=total,
                        total_presupuesto=total_presupuesto,
                        grafico_data=grafico_data,
                        no_presupuestos=no_presupuestos)


@app.route('/registrar', methods=["GET", "POST"])
def registrar():
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    # Obtener presupuestos asignados
    pres_df = pd.read_sql_query("SELECT categoria FROM presupuestos", conn)
    categorias_con_presupuesto = set(pres_df['categoria'])

    conn.close()

    if request.method == "POST":
        categoria = request.form['categoria']

        # Validación: si la categoría no tiene presupuesto asignado
        if categoria not in categorias_con_presupuesto:
            mensaje_error = f'No puedes registrar un gasto en {categoria} porque no tiene <a href="/presupuestos" style="color:white; text-decoration:underline;">presupuesto</a> asignado.' 

            return render_template("registro.html", mensaje_error=mensaje_error)

        # Si tiene presupuesto, registrar normalmente
        fecha = request.form['fecha']
        monto = float(request.form['monto'])
        comentario = request.form['comentario']

        conn = sqlite3.connect(DB)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO gastos (fecha, categoria, monto, comentario) VALUES (?, ?, ?, ?)",
                       (fecha, categoria, monto, comentario))
        conn.commit()
        conn.close()

        return redirect('/')

    return render_template("registro.html")


@app.route('/prediccion', methods=["GET", "POST"])
def prediccion():
    pred = ""
    grafico = ""
    categorias = ["Comida", "Transporte", "Ocio", "Salud", "Educación", "Paseos/Salidas", "Otros"]

    conn = sqlite3.connect(DB)

    # 1. Obtener DataFrame general siempre
    df_general = pd.read_sql_query("SELECT fecha, monto FROM gastos", conn)

    pred_general = ""
    grafico_general = ""

    if not df_general.empty and len(df_general) >= 2:
        # Procesamiento general
        df_general['fecha'] = pd.to_datetime(df_general['fecha'])
        df_general['dias'] = (df_general['fecha'] - df_general['fecha'].min()).dt.days
        Xg = df_general[['dias']]
        yg = df_general['monto']

        # Modelo regresión lineal
        modelo_g = LinearRegression()
        modelo_g.fit(Xg, yg)

        # Predicción para 7 días más
        futuro_dia_g = df_general['dias'].max() + 7
        futuro_g = pd.DataFrame({'dias': [futuro_dia_g]})
        prediccion_val_g = modelo_g.predict(futuro_g)[0]
        prediccion_val_g = max(prediccion_val_g, 0)  # No negativos
        pred_general = f"Gasto estimado general en 7 días más: ${prediccion_val_g:.2f}"

        # Gráfico general
        if not os.path.exists('static'):
            os.makedirs('static')

        ruta_general = "static/prediccion_general.png"

        plt.figure(figsize=(8,4))
        plt.scatter(df_general['fecha'], yg, color='blue', label='Datos históricos')

        fechas_pred_g = pd.date_range(start=df_general['fecha'].min(), end=df_general['fecha'].max())
        dias_pred_g = (fechas_pred_g - df_general['fecha'].min()).days.values.reshape(-1,1)
        y_pred_g = modelo_g.predict(dias_pred_g)
        plt.plot(fechas_pred_g, y_pred_g, color='green', label='Tendencia actual')

        fecha_futuro_g = df_general['fecha'].max() + pd.Timedelta(days=7)
        plt.scatter([fecha_futuro_g], [prediccion_val_g], color='red', label='Predicción (7 días más)')

        plt.title("Tendencia y Predicción General")
        plt.xlabel("Fecha")
        plt.ylabel("Monto")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(ruta_general)
        plt.close()
        grafico_general = ruta_general
    else:
        pred_general = "No hay suficientes datos para predecir."

    # Determina si mostrar formulario por categoría
    mostrar_formulario = not (df_general.empty or len(df_general) < 2)

    # 2. Si POST (filtro por categoría)
    if request.method == "POST" and mostrar_formulario:
        categoria = request.form['categoria']

        df = pd.read_sql_query("SELECT fecha, monto FROM gastos WHERE categoria = ?", conn, params=(categoria,))
        conn.close()

        if df.empty:
            pred = f"No hay datos para la categoría {categoria}."
        elif len(df) < 2:
            pred = f"No hay suficientes datos para predecir en la categoría {categoria}. Se necesitan al menos 2 registros."
        else:
            # Procesamiento por categoría
            df['fecha'] = pd.to_datetime(df['fecha'])
            df['dias'] = (df['fecha'] - df['fecha'].min()).dt.days
            X = df[['dias']]
            y = df['monto']

            modelo = LinearRegression()
            modelo.fit(X, y)

            futuro_dia = df['dias'].max() + 7
            futuro = pd.DataFrame({'dias': [futuro_dia]})
            prediccion_val = modelo.predict(futuro)[0]

            if prediccion_val <= 0:
                suma_montos = df['monto'].sum()
                prediccion_val = suma_montos / 1.7

            pred = f"Gasto estimado en 7 días más para {categoria}: ${prediccion_val:.2f}"


            # Gráfico por categoría
            if not os.path.exists('static'):
                os.makedirs('static')

            nombre_categoria_archivo = categoria.replace("/", "_")
            ruta = f"static/prediccion_{nombre_categoria_archivo}.png"

            plt.figure(figsize=(8,4))
            plt.scatter(df['fecha'], y, color='blue', label='Datos históricos')

            fechas_pred = pd.date_range(start=df['fecha'].min(), end=df['fecha'].max())
            dias_pred = (fechas_pred - df['fecha'].min()).days.values.reshape(-1,1)
            y_pred = modelo.predict(dias_pred)
            plt.plot(fechas_pred, y_pred, color='green', label='Tendencia actual')

            fecha_futuro = df['fecha'].max() + pd.Timedelta(days=7)
            plt.scatter([fecha_futuro], [prediccion_val], color='red', label='Predicción (7 días más)')

            plt.title(f"Tendencia y Predicción - {categoria}")
            plt.xlabel("Fecha")
            plt.ylabel("Monto")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(ruta)
            plt.close()
            grafico = ruta
    else:
        conn.close()

    return render_template("prediccion.html",
                           pred_general=pred_general,
                           grafico_general=grafico_general,
                           prediccion=pred,
                           categorias=categorias,
                           grafico=grafico,
                           mostrar_formulario=mostrar_formulario)


@app.route('/eliminar/<int:id>')
def eliminar(id):
    mes = request.args.get('mes')
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM gastos WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    if mes:
        return redirect(f'/historico?mes={mes}')
    return redirect(request.referrer or '/')


@app.route('/historico', methods=["GET", "POST"])
def historico():
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM gastos", conn)
    pres_df = pd.read_sql_query("SELECT * FROM presupuestos", conn)
    conn.close()

    gastos = []
    mensaje = ""
    resumen_semanal = []
    resumen_mensual = []
    alerts = []
    mes_seleccionado = None

    if not df.empty:
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['fecha'] = df['fecha'].dt.strftime('%d-%m-%Y')  # Formatear directamente
        df['semana'] = pd.to_datetime(df['fecha'], format='%d-%m-%Y').dt.isocalendar().week
        df['mes'] = pd.to_datetime(df['fecha'], format='%d-%m-%Y').dt.month

    if df.empty:
        mensaje = "No hay datos para mostrar."

    # GET: filtro con query string ?mes=
    if request.method == "GET" and 'mes' in request.args:
        mes = request.args.get('mes')
        mes_seleccionado = mes

        if mes:
            df['fecha_dt'] = pd.to_datetime(df['fecha'], format='%d-%m-%Y')
            filtrado = df[df['fecha_dt'].dt.month == int(mes)]
            filtrado = filtrado.sort_values(by='fecha_dt', ascending=False)
            gastos = filtrado.drop(columns='fecha_dt').to_dict(orient="records")
            if not gastos:
                mensaje = "No hay gastos registrados para este mes."

    # POST: filtro desde el formulario
    elif request.method == "POST":
        mes = request.form['mes']
        mes_seleccionado = mes

        df['fecha_dt'] = pd.to_datetime(df['fecha'], format='%d-%m-%Y')
        filtrado = df[df['fecha_dt'].dt.month == int(mes)]
        filtrado = filtrado.sort_values(by='fecha_dt', ascending=False)
        gastos = filtrado.drop(columns='fecha_dt').to_dict(orient="records")
        if not gastos:
            mensaje = "No hay gastos registrados para este mes."

    return render_template("historico.html",
                           gastos=gastos,
                           mensaje=mensaje,
                           hay_datos=not df.empty,
                           resumen_semanal=resumen_semanal,
                           resumen_mensual=resumen_mensual,
                           mes_seleccionado=mes_seleccionado,
                           alerts=alerts)


@app.route('/presupuestos', methods=["GET", "POST"])
def presupuestos():
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    
    if request.method == "POST":
        categoria = request.form['categoria']
        monto_max = float(request.form['monto_max'])
        cursor.execute("INSERT OR REPLACE INTO presupuestos (categoria, monto_max) VALUES (?, ?)", (categoria, monto_max))
        conn.commit()

    # 🔥 Reconsultar presupuestos después del commit
    cursor.execute("SELECT * FROM presupuestos")
    presupuestos = cursor.fetchall()
    conn.close()

    categorias = ["Comida", "Transporte", "Ocio", "Salud", "Educación", "Paseos/Salidas", "Otros"]

    # Convertir presupuestos a dict
    presupuestos_dict = {p[0]: p[1] for p in presupuestos}

    conn = sqlite3.connect(DB)
    gastos_df = pd.read_sql_query("SELECT categoria, SUM(monto) as gastado FROM gastos GROUP BY categoria", conn)
    conn.close()

    utilizado_dict = dict(zip(gastos_df['categoria'], gastos_df['gastado']))

    # Generar alertas de presupuesto
    alertas = []

    for cat in categorias:
        if presupuestos_dict.get(cat):
            presupuesto = presupuestos_dict[cat]
            gastado = utilizado_dict.get(cat, 0)

            if gastado >= presupuesto:
                alertas.append(f"❌ Has sobrepasado tu presupuesto en {cat}. Gastaste ${gastado:.2f} de ${presupuesto:.2f}.")
            elif gastado >= 0.8 * presupuesto:
                alertas.append(f"⚠️ Estás cerca de tu presupuesto en {cat}. Gastaste ${gastado:.2f} de ${presupuesto:.2f}.")

    
    return render_template("presupuestos.html", presupuestos=presupuestos_dict, categorias=categorias, utilizado=utilizado_dict, alertas=alertas)



if __name__ == '__main__':
    app.run(debug=True)
