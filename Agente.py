from flask import Flask, render_template, request, redirect, session

import sqlite3
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = "."  # Necesario para usar session
DB = "gastos.db"

# ======================= CREAR TABLAS =======================

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
    monto_max REAL,
    monto_ahorro REAL DEFAULT 0
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS metas_generales (
    meta TEXT PRIMARY KEY,
    valor REAL
)
""")

conn.commit()
conn.close()

# ======================= BDI MODULE =======================

class Beliefs:
    def __init__(self, db):
        conn = sqlite3.connect(db)
        self.df = pd.read_sql_query("SELECT * FROM gastos", conn)
        conn.close()

    def gastos_por_categoria(self):
        if self.df.empty:
            return {}
        return self.df.groupby('categoria')['monto'].sum().to_dict()

    def gasto_total(self):
        return self.df['monto'].sum() if not self.df.empty else 0

    def ahorro_por_categoria(self):
        conn = sqlite3.connect(DB)
        df = pd.read_sql_query("SELECT categoria, monto_ahorro FROM presupuestos", conn)
        conn.close()
        return df.set_index('categoria')['monto_ahorro'].to_dict()

class Desires:
    def __init__(self, db):
        self.metas = {}
        conn = sqlite3.connect(db)
        cursor = conn.cursor()

        # Leer metas generales
        cursor.execute("SELECT meta, valor FROM metas_generales")
        for meta, valor in cursor.fetchall():
            self.metas[meta] = valor

        # Leer presupuestos (max) por categoría
        cursor.execute("SELECT categoria, monto_max FROM presupuestos")
        for categoria, monto_max in cursor.fetchall():
            self.metas[f"{categoria}_max"] = monto_max

        conn.close()

class Intentions:
    def __init__(self, beliefs, desires):
        self.beliefs = beliefs
        self.desires = desires

    def planificar(self):
        planes = []
        gastos = self.beliefs.gastos_por_categoria()

        # 🔥 Diccionario para agrupar categorías por tipo de mensaje
        agrupados = {
            'sobrepasado': [],
            'usado_todo': [],
            'cerca': []
        }

        for cat, gasto in gastos.items():
            maximo = self.desires.metas.get(f"{cat}_max", None)
            if maximo:
                if gasto > maximo:
                    agrupados['sobrepasado'].append((cat, gasto, maximo))
                elif gasto == maximo:
                    agrupados['usado_todo'].append(cat)
                elif gasto >= 0.8 * maximo:
                    agrupados['cerca'].append((cat, gasto, maximo))

        # 🔧 Generar mensajes agrupados

        if agrupados['sobrepasado']:
            cats = ', '.join([f"{cat} (${gasto:.2f} / ${maximo:.2f})" for cat, gasto, maximo in agrupados['sobrepasado']])
            planes.append(f"❌ Has sobrepasado tu presupuesto en: {cats}.")

        if agrupados['usado_todo']:
            cats = ', '.join(agrupados['usado_todo'])
            planes.append(f"❌ Usaste todo el presupuesto en: {cats}.")

        if agrupados['cerca']:
            cats = ', '.join([f"{cat}"])
            planes.append(f"⚠️ Estás cerca del límite en {cats}.")

        # 🔧 Agregar mensaje de ahorro mensual
        ahorro_objetivo = self.desires.metas.get("ahorro_mensual", None)
        if ahorro_objetivo:
            gasto_total = self.beliefs.gasto_total()
            planes.append(f"💡 Revisa tu ahorro mensual. Gastado total: ${gasto_total:.2f}, Objetivo ahorro: ${ahorro_objetivo:.2f}.")

        # 🔧 Si no hay planes, mensaje positivo general
        if not planes:
            planes.append("✅ Todo en orden. ¡Buen manejo de tus gastos!")

        return planes

    def reasignar_presupuestos(self):
        conn = sqlite3.connect(DB)
        cursor = conn.cursor()

        # Obtener presupuestos y gastos actuales
        cursor.execute("SELECT categoria, monto_max FROM presupuestos")
        presupuestos = cursor.fetchall()
        presupuesto_dict = {p[0]: p[1] for p in presupuestos}

        gastos = self.beliefs.gastos_por_categoria()

        # Detectar categorías sobrepasadas y calcular exceso
        excesos = {}
        for cat, gasto in gastos.items():
            maximo = presupuesto_dict.get(cat)
            if maximo and gasto > maximo:
                excesos[cat] = gasto - maximo

        total_exceso = sum(excesos.values())

        if total_exceso == 0:
            conn.close()
            return "✅ No hay excesos para reasignar."

        # 🔥 Filtrar categorías no sobrepasadas y con presupuesto liberado
        disponibles = []
        for cat in presupuesto_dict:
            if cat not in excesos:
                gasto_cat = gastos.get(cat, 0)
                maximo_cat = presupuesto_dict[cat]
                if gasto_cat < maximo_cat:
                    disponibles.append(cat)

        if not disponibles:
            conn.close()
            return "⚠️ No hay categorías con presupuesto disponible para reasignar."

        # Calcular reduccion proporcional
        reduccion_por_categoria = total_exceso / len(disponibles)

        for cat in disponibles:
            nuevo_presupuesto = presupuesto_dict[cat] - reduccion_por_categoria
            nuevo_presupuesto = max(nuevo_presupuesto, 0)  # evitar negativos
            cursor.execute("UPDATE presupuestos SET monto_max = ? WHERE categoria = ?", (nuevo_presupuesto, cat))

        # 🔥 Sumar el exceso total a la categoría sobrepasada más prioritaria (ejemplo: la primera)
        cat_prioritaria = list(excesos.keys())[0]
        presupuesto_actual = presupuesto_dict[cat_prioritaria]
        nuevo_presupuesto = presupuesto_actual + total_exceso
        cursor.execute("UPDATE presupuestos SET monto_max = ? WHERE categoria = ?", (nuevo_presupuesto, cat_prioritaria))

        conn.commit()
        conn.close()

        return f"🔄 Se reasignó un total de ${total_exceso:.2f}, reduciendo ${reduccion_por_categoria:.2f} en cada categoría disponible y sumándolo a {cat_prioritaria}."


# ======================= ROUTES =======================

# @app.route('/planificador')
# def planificador():
#     beliefs = Beliefs(DB)
#     desires = Desires(DB)
#     intentions = Intentions(beliefs, desires)
#     planes = intentions.planificar()
#     return render_template("planificador.html", planes=planes)

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
    pres_df = pd.read_sql_query("SELECT categoria FROM presupuestos", conn)
    conn.close()

    categorias_con_presupuesto = pres_df['categoria'].tolist()


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

    return render_template("registro.html",
                       categorias=categorias_con_presupuesto,
                       mensaje_error=mensaje_error if 'mensaje_error' in locals() else None)

@app.route('/prediccion', methods=["GET", "POST"])
def prediccion():
    pred = ""
    grafico = ""
    conn = sqlite3.connect(DB)
    categorias_df = pd.read_sql_query("SELECT DISTINCT categoria FROM presupuestos", conn)
    conn.close()
    categorias = categorias_df['categoria'].tolist()

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

    # ✅ Obtener meses con datos registrados
    meses_disponibles_df = pd.read_sql_query("""
        SELECT DISTINCT strftime('%m', fecha) AS mes
        FROM gastos
        ORDER BY mes
    """, conn)
    meses_disponibles = meses_disponibles_df['mes'].astype(int).tolist()

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

    mostrar_cartola = False
    if gastos:
        mostrar_cartola = True

    return render_template("historico.html",
                           gastos=gastos,
                           mensaje=mensaje,
                           hay_datos=not df.empty,
                           resumen_semanal=resumen_semanal,
                           resumen_mensual=resumen_mensual,
                           mes_seleccionado=mes_seleccionado,
                           alerts=alerts,
                           mostrar_cartola=mostrar_cartola,
                           meses_disponibles=meses_disponibles)

@app.route('/presupuestos', methods=["GET", "POST"])
def presupuestos():

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    if request.method == "POST":
        categoria = request.form['categoria']
        monto_max = float(request.form['monto_max'])
        monto_ahorro = float(request.form.get('monto_ahorro', 0))  # ✅ Nuevo campo de ahorro
        cursor.execute("""
            INSERT OR REPLACE INTO presupuestos (categoria, monto_max, monto_ahorro)
            VALUES (?, ?, ?)
        """, (categoria, monto_max, monto_ahorro))
        conn.commit()

    conn.close()

    # ======================= BDI: beliefs, desires, intentions =======================
    beliefs = Beliefs(DB)
    desires = Desires(DB)
    intentions = Intentions(beliefs, desires)
    planes = intentions.planificar()

    # ======================= Reasignación automática de presupuestos =======================
    reasignacion_resultado = intentions.reasignar_presupuestos()

    # 🔥 Si ambos son mensajes positivos, combina en uno solo
    if planes == ["✅ Todo en orden. ¡Buen manejo de tus gastos!"] and "No hay excesos para reasignar" in reasignacion_resultado:
        planes = ["✅ Todo en orden. ¡Buen manejo de tus gastos y presupuestos!"]
    else:
        planes.append(reasignacion_resultado)

    # 🔥 Leer nuevamente presupuestos actualizados después de la reasignación
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("SELECT categoria, monto_max, monto_ahorro FROM presupuestos")
    presupuestos = cursor.fetchall()
    conn.close()

    # Convertir a dicts separados
    presupuestos_dict = {p[0]: p[1] for p in presupuestos}
    presupuestos_ahorro = {p[0]: p[2] for p in presupuestos}

    # ✅ Construir lista de categorias
    categorias = list(presupuestos_dict.keys())

    # Leer utilizado actualizado
    conn = sqlite3.connect(DB)
    gastos_df = pd.read_sql_query("SELECT categoria, SUM(monto) as gastado FROM gastos GROUP BY categoria", conn)
    conn.close()
    utilizado_dict = dict(zip(gastos_df['categoria'], gastos_df['gastado']))

    # Generar alertas actualizadas
    alertas = []
    for cat in categorias:
        presupuesto = presupuestos_dict[cat]
        gastado = utilizado_dict.get(cat, 0)
        if gastado > presupuesto:
            alertas.append(f"❌ Has sobrepasado tu presupuesto en {cat}. Gastaste ${gastado:.2f} de ${presupuesto:.2f}.")
        elif gastado == presupuesto:
            alertas.append(f"⚠️ Usaste todo tu presupuesto en {cat}. Gastaste ${gastado:.2f} de ${presupuesto:.2f}.")
        elif gastado >= 0.8 * presupuesto:
            alertas.append(f"⚠️ Estás cerca de tu presupuesto en {cat}. Gastaste ${gastado:.2f} de ${presupuesto:.2f}.")

    planes = list(dict.fromkeys(planes))
    return render_template("presupuestos.html",
                           categorias=categorias,
                           presupuestos=presupuestos_dict,
                           presupuestos_ahorro=presupuestos_ahorro,
                           utilizado=utilizado_dict,
                           alertas=alertas,
                           planes=planes)

@app.route('/presupuestos/agregar', methods=["POST"])
def agregar_presupuesto():
    categoria = request.form['nueva_categoria']
    monto_max = float(request.form['nuevo_monto_max'])
    monto_ahorro = float(request.form.get('nuevo_monto_ahorro', 0))  # ✅ Captura el ahorro

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO presupuestos (categoria, monto_max, monto_ahorro)
        VALUES (?, ?, ?)
    """, (categoria, monto_max, monto_ahorro))
    conn.commit()
    conn.close()

    return redirect('/presupuestos')

@app.route('/presupuestos/eliminar/<categoria>')
def eliminar_presupuesto(categoria):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    # 🔥 Primero, eliminar los gastos asociados a la categoría
    cursor.execute("DELETE FROM gastos WHERE categoria = ?", (categoria,))

    # Luego, eliminar la categoría en presupuestos
    cursor.execute("DELETE FROM presupuestos WHERE categoria = ?", (categoria,))
    
    conn.commit()
    conn.close()
    return redirect('/presupuestos')

@app.route('/cartola/<int:mes>')
def cartola(mes):
    import calendar
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    import sqlite3
    import pandas as pd

    # ✅ Lista de meses en español
    meses_es = ["", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM gastos", conn)
    conn.close()

    if df.empty:
        return "No hay datos registrados."

    df['fecha'] = pd.to_datetime(df['fecha'])
    df_mes = df[df['fecha'].dt.month == mes]

    if df_mes.empty:
        return "No hay gastos registrados para este mes."

    # Día de mayor gasto
    df_mes['dia'] = df_mes['fecha'].dt.day
    gastos_por_dia = df_mes.groupby('dia')['monto'].sum()
    dia_max_gasto = gastos_por_dia.idxmax()
    monto_max_gasto = gastos_por_dia.max()

    # ✅ Ordenar dataframe por fecha ascendente antes de crear el PDF
    df_mes = df_mes.sort_values(by='fecha', ascending=True)

    # Total gastado en el mes
    total_mes = df_mes['monto'].sum()

    # Gráfico lineal de gastos diarios
    plt.figure(figsize=(10,5))
    ax = gastos_por_dia.plot(kind='line', marker='o', color='green')
    plt.title(f"Gastos diarios - {meses_es[mes]}")
    plt.xlabel("Día")
    plt.ylabel("Monto gastado")
    ax.set_xticks(gastos_por_dia.index)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    grafico_path = f"static/cartola_mes_{mes}.png"
    plt.savefig(grafico_path)
    plt.close()

    # Generar PDF
    pdf = FPDF()
    pdf.add_page()

    # Título
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, f"Cartola Mensual de {meses_es[mes]}", ln=1, align='C')
    pdf.set_font("Arial", '', 14)
    # pdf.cell(0, 10, f"Mes: {meses_es[mes]}", ln=1, align='C')
    pdf.ln(5)
    pdf.set_draw_color(0, 128, 0)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)

    # Detalle de gastos (tabla)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Detalle de gastos", ln=1)
    pdf.set_draw_color(169, 169, 169)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Centrar tabla
    x_inicio = (210 - 145) / 2

    # Cabecera tabla
    pdf.set_x(x_inicio)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(35, 8, "Categoría", border=1, align='C')
    pdf.cell(60, 8, "Comentario", border=1, align='C')
    pdf.cell(25, 8, "Fecha", border=1, align='C')
    pdf.cell(25, 8, "Monto", border=1, align='C')
    pdf.ln()

    # Filas de la tabla
    pdf.set_font("Arial", '', 10)
    for idx, row in df_mes.iterrows():
        pdf.set_x(x_inicio)
        comentario = str(row['comentario'])
        if len(comentario) > 35:
            comentario = comentario[:32] + "..."

        pdf.cell(35, 8, str(row['categoria']), border=1)
        pdf.cell(60, 8, comentario, border=1)
        pdf.cell(25, 8, row['fecha'].strftime('%d-%m-%Y'), border=1)
        pdf.cell(25, 8, f"${row['monto']:.2f}", border=1)
        pdf.ln()

    # Fila de total general
    pdf.set_font("Arial", 'B', 10)
    pdf.set_x(x_inicio)
    pdf.cell(120, 8, "TOTAL", border=1, align='R')
    pdf.cell(25, 8, f"${total_mes:.2f}", border=1, align='C')
    pdf.ln()

    # Gráfico
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Gráfico de gastos diarios", ln=1)
    pdf.image(grafico_path, x=15, y=None, w=pdf.w - 30)
    pdf.ln(5)

    # Guardar PDF
    pdf_output = f"static/cartola_mes_{mes}.pdf"
    pdf.output(pdf_output)

    return redirect(f"/{pdf_output}")


if __name__ == '__main__':
    app.run(debug=True)
