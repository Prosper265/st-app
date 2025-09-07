import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import re
import unicodedata
import base64

# Set page config
st.set_page_config(page_title="Malawi Food Policy Simulator", layout="wide", page_icon="🌍")

# App color palette
PALETTE = {
    "primary_dark": "#123024",
    "primary_darker": "#081c15",
    "primary_light": "#74c69d",
    "background_tint": "#d8f3dc",
}

# Detect theme (light/dark) to adapt charts
try:
    IS_DARK = st.get_option("theme.base") == "dark"
except Exception:
    IS_DARK = False

def get_plotly_template():
    return "plotly_dark" if IS_DARK else "plotly_white"

# Function to display Malawi flag
def display_flag():
    """Display the Malawi flag in the header"""
    try:
        with open("data/malawi/flag.png", "rb") as f:
            flag_data = base64.b64encode(f.read()).decode()
        
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{flag_data}" 
                     style="height: 60px; width: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            </div>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("Malawi flag image not found")

# Function to set background image with better text visibility
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        :root {{
            --heading: {PALETTE["primary_dark"]};
            --text: {PALETTE["primary_darker"]};
            --button-bg: {PALETTE["primary_light"]};
            --button-text: {PALETTE["primary_darker"]};
            --card-bg: rgba(216, 243, 220, 0.92);
        }}
        /* Streamlit theme-driven override */
        [data-base-theme="dark"] :root {{
            --heading: {PALETTE["background_tint"]};
            --text: #e6fff0;
            --button-bg: {PALETTE["primary_light"]};
            --button-text: {PALETTE["primary_darker"]};
            --card-bg: rgba(8, 28, 21, 0.6);
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --heading: {PALETTE["background_tint"]};
                --text: #e6fff0;
                --button-bg: {PALETTE["primary_light"]};
                --button-text: {PALETTE["primary_darker"]};
                --card-bg: rgba(8, 28, 21, 0.6);
            }}
        }}
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: var(--text);
        }}
        .main .block-container {{
            background-color: var(--card-bg);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 6px 14px rgba(0, 0, 0, 0.12);
            color: var(--text);
        }}
        /* Ensure headings are always visible */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: var(--heading) !important;
            text-shadow: 0 1px 1px rgba(0,0,0,0.25);
        }}
        .stMetric {{
            background-color: rgba(0, 0, 0, 0.6);
            border-radius: 8px;
            padding: 10px;
            border: 1px solid {PALETTE["primary_light"]}44;
            color: white !important;
        }}
        .stDataFrame {{
            background-color: transparent;
        }}
        .stRadio > div {{
            background-color: transparent;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid {PALETTE["primary_light"]}44;
        }}
        .stSelectbox, .stMultiselect, .stSlider {{
            background-color: transparent;
        }}
        /* Sidebar styling */
        section[data-testid="stSidebar"] > div {{
            background: linear-gradient(180deg, {PALETTE["primary_darker"]} 0%, {PALETTE["primary_dark"]} 100%);
        }}
        section[data-testid="stSidebar"] * {{ color: #f0fff4 !important; }}
        /* Sidebar radio navigation styling */
        .stRadio [role="radiogroup"] > label {{ display: block; margin-bottom: 8px; }}
        .stRadio [role="radiogroup"] {{ display: flex; flex-direction: column; gap: 10px; }}
        .stRadio [role="radiogroup"] [role="radio"] {{
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 10px;
            padding: 10px 12px;
            color: #f0fff4 !important;
        }}
        .stRadio [role="radiogroup"] [role="radio"][aria-checked="true"] {{
            background: {PALETTE["primary_light"]};
            color: #0b0b0b !important;
            border-color: {PALETTE["background_tint"]};
            font-weight: 700;
        }}
        .stRadio [role="radiogroup"] [role="radio"] * {{ color: inherit !important; }}
        section[data-testid="stSidebar"] {{ padding-top: 12px; }}
        /* Buttons */
        .stButton > button {{
            background-color: var(--button-bg);
            color: var(--button-text);
            border: none;
            border-radius: 6px;
        }}
        .stButton > button:hover {{
            filter: brightness(0.95);
        }}
        /* General text */
        .main p, .main label, .main span, .main li, .main div, .main code {{ color: var(--text); }}
        /* High-contrast labels for select-like widgets */
        div[data-testid="stSelectbox"] > label,
        div[data-testid="stMultiSelect"] > label,
        div[data-testid="stSlider"] > label,
        div[data-testid="stRadio"] > label {{
            color: {PALETTE["primary_darker"]} !important;
            background: {PALETTE["background_tint"]};
            padding: 2px 8px;
            border-radius: 8px;
            display: inline-block;
            margin-bottom: 6px;
            text-shadow: 0 1px 0 rgba(255,255,255,0.6);
        }}
        [data-base-theme="dark"] div[data-testid="stSelectbox"] > label,
        [data-base-theme="dark"] div[data-testid="stMultiSelect"] > label,
        [data-base-theme="dark"] div[data-testid="stSlider"] > label,
        [data-base-theme="dark"] div[data-testid="stRadio"] > label {{
            color: #0b0b0b !important;
        }}
        /* Select boxes styling */
        .main div[data-baseweb="select"] > div {{
            background: rgba(255,255,255,0.1);
            border: 1px solid {PALETTE["primary_light"]};
            border-radius: 10px;
            color: #0b0b0b !important;
        }}
        .main div[data-baseweb="select"] input, .main div[data-baseweb="select"] * {{
            color: #0b0b0b !important;
        }}
        .main div[data-baseweb="select"]:hover > div {{
            box-shadow: 0 0 0 2px {PALETTE["background_tint"]};
        }}
        /* Ensure main radios are visible */
        .main .stRadio [role="radiogroup"] {{ display: flex; flex-direction: column; gap: 8px; }}
        .main .stRadio [role="radiogroup"] [role="radio"] {{
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 999px;
            padding: 8px 14px;
            color: #0b0b0b !important;
            display: inline-flex;
            margin-right: 8px;
        }}
        .main .stRadio [role="radiogroup"] [role="radio"][aria-checked="true"] {{
            background: {PALETTE["background_tint"]};
            color: #0b0b0b !important;
            border-color: {PALETTE["primary_light"]};
            font-weight: 700;
        }}
        .main .stRadio [role="radiogroup"] [role="radio"] * {{ color: inherit !important; }}
        
        /* Intervention type buttons styling */
        .stButton > button {{
            border-radius: 8px;
            padding: 8px 12px;
+            margin: 2px;
+            transition: all 0.2s ease;
         }}
+        /* Active intervention button */
+        .stButton:has(button[id*="intervention"]) > button {{
+            background: {PALETTE["background_tint"]};
+            color: {PALETTE["primary_darker"]};
+            border: 1px solid {PALETTE["primary_light"]};
+            font-weight: 600;
+        }}
+        .stButton:has(button[id*="intervention"]) > button:hover {{
+            background: {PALETTE["primary_light"]};
+            color: {PALETTE["primary_darker"]};
+        }}

        /* Per-option active colors for radio groups (e.g., Select Dataset) */
        .main .stRadio [role="radiogroup"] [role="radio"]:nth-child(1)[aria-checked="true"] {{ background: {PALETTE["background_tint"]}; }}
        .main .stRadio [role="radiogroup"] [role="radio"]:nth-child(2)[aria-checked="true"] {{ background: {PALETTE["primary_light"]}; }}
        .main .stRadio [role="radiogroup"] [role="radio"]:nth-child(3)[aria-checked="true"] {{ background: #b8e3cc; }}
        .main .stRadio [role="radiogroup"] [role="radio"]:nth-child(4)[aria-checked="true"] {{ background: #a1dcc0; }}
        .main .stRadio [role="radiogroup"] [role="radio"]:nth-child(5)[aria-checked="true"] {{ background: #8fd6b8; }}

        /* Tabs color coding and active state */
        .stTabs [role="tablist"] {{ gap: 6px; margin-top: 8px; margin-bottom: 10px; }}
        .stTabs [role="tablist"] [role="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 10px 16px;
            margin-right: 0;
            border: 1px solid rgba(0,0,0,0.05);
            background: rgba(255,255,255,0.7);
            color: {PALETTE["primary_darker"]};
        }}
        .stTabs [role="tablist"] [role="tab"][aria-selected="true"] {{
            color: #0b0b0b;
            box-shadow: 0 2px 0 0 {PALETTE["primary_light"]} inset;
        }}
        /* Per-tab accents by index */
        .stTabs [role="tablist"] [role="tab"]:nth-child(1)[aria-selected="true"] {{ background: {PALETTE["background_tint"]}; }}
        .stTabs [role="tablist"] [role="tab"]:nth-child(2)[aria-selected="true"] {{ background: {PALETTE["primary_light"]}; }}
        .stTabs [role="tablist"] [role="tab"]:nth-child(3)[aria-selected="true"] {{ background: #b8e3cc; }}
        .stTabs [role="tablist"] [role="tab"]:nth-child(4)[aria-selected="true"] {{ background: #a1dcc0; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image (assuming su.jpg is in the same directory)
try:
    set_background("su.jpg")
except FileNotFoundError:
    st.warning("Background image 'su.jpg' not found. Using default background.")

# Get the data directory path
DATA_PATH = Path("data/malawi")

# ----- Data Loading -----
@st.cache_data
def load_data():
    try:
        # Load food composition data - use food_composition.csv (better data quality)
        food_comp = pd.read_csv(DATA_PATH / "food_composition.csv")
        
        # Load consumption data
        consumption = pd.read_csv(DATA_PATH / "initial_cons.csv")
        
        # Load additional Malawi-specific data
        adequacy_df = pd.read_csv(DATA_PATH / "nutrient_adequacy.csv")
        gender_df = pd.read_csv(DATA_PATH / "gender_comparison.csv")
        simulations_df = pd.read_csv(DATA_PATH / "simulations.csv")
        
        return food_comp, consumption, adequacy_df, gender_df, simulations_df
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.info("Please make sure all CSV files are in the data/malawi/ directory")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

food_comp, consumption, adequacy_df, gender_df, simulations_df = load_data()

# Check if data loaded successfully
if food_comp.empty or consumption.empty:
    st.stop()

# ----- Data Preprocessing -----
def preprocess_data(food_comp, consumption):
    # Clean food composition data
    food_comp.columns = [col.strip() for col in food_comp.columns]
    
    # Clean food names - remove numbers after commas
    if 'Food_name' in food_comp.columns:
        food_comp['Food_name'] = food_comp['Food_name'].apply(
            lambda x: re.sub(r',\s*\d+.*', '', str(x)) if pd.notnull(x) else x
        )
    
    # Create user-friendly nutrient names mapping
    nutrient_mapping = {
        'E_kcal': 'Energy (kcal)',
        'Prot_g': 'Protein (g)',
        'RAE_μg': 'Vitamin A (μg)',
        'VitC_mg': 'Vitamin C (mg)',
        'Thiamin_mg': 'Thiamin (mg)',
        'Riboflavin_mg': 'Riboflavin (mg)',
        'Nia_mg': 'Niacin (mg)',
        'Fol_μg': 'Folate (μg)',
        'VitB12_μg': 'Vitamin B12 (μg)',
        'Ca_mg': 'Calcium (mg)',
        'Fe_mg': 'Iron (mg)',
        'Zn_mg': 'Zinc (mg)'
    }
    
    # Rename nutrient columns to be more user-friendly
    food_comp = food_comp.rename(columns=nutrient_mapping)
    
    # Extract nutrient information from food composition (exclude Food_code and Food_name)
    nutrients = [col for col in food_comp.columns if col not in ['Food_code', 'Food_name']]
    
    # Process consumption data - filter for Malawi only
    malawi_consumption = consumption[consumption['Area'] == 'Malawi'].copy()
    malawi_consumption['Item'] = malawi_consumption['Item'].str.title()
    
    # Clean food names in consumption data too
    if 'Item' in malawi_consumption.columns:
        malawi_consumption['Item'] = malawi_consumption['Item'].apply(
            lambda x: re.sub(r',\s*\d+.*', '', str(x)) if pd.notnull(x) else x
        )
    
    return food_comp, malawi_consumption, nutrients

food_comp, malawi_consumption, nutrients = preprocess_data(food_comp, consumption)

# ----- Nutrient Standardization -----
def standardize_nutrient_name(name):
    if pd.isnull(name): return ""
    name = name.lower().strip()
    # Remove common prefixes and clean up
    name = re.sub(r'^consumption-adequacy-of\s*', '', name)
    name = re.sub(r'\([^)]*\)', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.title()

# Standardize column names in adequacy data
if not adequacy_df.empty:
    adequacy_df.columns = [standardize_nutrient_name(col) for col in adequacy_df.columns]

# ----- UI Organization -----
def dashboard_section(title, explanation):
    st.markdown(f"### {title}")
    with st.expander(f"ℹ️ {title} Details"):
        st.write(explanation)

# ----- Visualization Functions -----
def render_metric_card(title: str, value: str, background_color: str, text_color: str = "#0b0b0b"):
    st.markdown(
        f"""
        <div style="background:{background_color};color:{text_color};padding:14px;border-radius:10px;border:1px solid rgba(0,0,0,0.06);">
            <div style="font-size:13px;opacity:0.9;margin-bottom:4px;">{title}</div>
            <div style="font-size:28px;font-weight:700;line-height:1;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# (Removed segmented_control helper; using radio styling instead)
def create_nutrient_radar_chart(sel_nutrients, radar_vals, district):
    fig = go.Figure(data=go.Scatterpolar(
        r=radar_vals,
        theta=sel_nutrients,
        fill='toself',
        name=district,
        line_color=PALETTE["primary_light"],
        fillcolor="rgba(116, 198, 157, 0.25)"
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
                     showlegend=True, template=get_plotly_template(), 
                     title=f"Nutrient Profile: {district}", height=400)
    return fig

def create_deficiency_heatmap(adequacy_df):
    nutrient_cols = [col for col in adequacy_df.columns if col != 'District']
    deficiency_data = [{'Nutrient': n, 'Deficiency Rate (%)': (len(adequacy_df[adequacy_df[n] < 80]) / len(adequacy_df)) * 100, 
                       'Severity': 'Critical' if rate > 50 else 'High' if rate > 30 else 'Moderate'} 
                      for n, rate in [(n, (len(adequacy_df[adequacy_df[n] < 80]) / len(adequacy_df)) * 100) for n in nutrient_cols]]
    deficiency_df = pd.DataFrame(deficiency_data).sort_values('Deficiency Rate (%)', ascending=False)
    fig = px.bar(deficiency_df, x='Nutrient', y='Deficiency Rate (%)', color='Severity', 
                 color_discrete_map={'Critical': '#B22222', 'High': '#FF8C00', 'Moderate': PALETTE["primary_light"]}, 
                 title="Nutrient Deficiency Rates Across Malawi Districts")
    fig.update_layout(template=get_plotly_template(), height=400)
    return fig

def create_interactive_scatter(adequacy_df, x_nutrient, y_nutrient):
    fig = px.scatter(
        adequacy_df, x=x_nutrient, y=y_nutrient, hover_name='District',
                     title=f"{x_nutrient} vs {y_nutrient} Adequacy in Malawi Districts", 
        color_discrete_sequence=[PALETTE["primary_light"]]
    )
    fig.update_layout(template=get_plotly_template(), height=500)
    return fig

def create_district_nutrient_heatmap(adequacy_df):
    if adequacy_df.empty:
        return go.Figure()
    heat_df = adequacy_df.set_index('District')
    fig = px.imshow(
        heat_df.T,
        color_continuous_scale=[[0, '#ffedea'], [0.5, PALETTE["primary_light"]], [1, PALETTE["primary_dark"]]],
        aspect='auto',
        labels=dict(color='Adequacy (%)'),
        title='District vs Nutrient Adequacy Heatmap'
    )
    fig.update_layout(template=get_plotly_template(), height=600)
    return fig

# ----- Intervention Simulation -----
class Intervention:
    def __init__(self, name, nutrient, efficacy, coverage):
        self.name = name
        self.nutrient = nutrient
        self.efficacy = efficacy
        self.coverage = coverage

def simulate_intervention(adequacy_df, intervention, district=None):
    df = adequacy_df.copy() if not district else adequacy_df[adequacy_df['District'] == district].copy()
    if intervention.nutrient in df.columns:
        current = df[intervention.nutrient]
        improvement = intervention.efficacy * intervention.coverage
        df[intervention.nutrient] = np.minimum(100, current + improvement)
    return df

def create_intervention_comparison(adequacy_df, interventions, district):
    baseline = adequacy_df[adequacy_df['District'] == district].melt(id_vars='District', var_name='Nutrient', value_name='Adequacy').assign(Scenario='Baseline')
    intervention_data = []
    for intervention in interventions:
        sim_data = simulate_intervention(adequacy_df, intervention, district).melt(id_vars='District', var_name='Nutrient', value_name='Adequacy').assign(Scenario=intervention.name)
        intervention_data.append(sim_data)
    plot_data = pd.concat([baseline] + intervention_data)
    fig = px.bar(plot_data, x='Nutrient', y='Adequacy', color='Scenario', barmode='group', 
                 title=f"Intervention Impacts: {district}")
    fig.update_layout(template="plotly_white", height=500)
    return fig

# ----- Main App -----
def main():
    # Display flag in sidebar
    try:
        with open("data/malawi/flag.png", "rb") as f:
            flag_data = base64.b64encode(f.read()).decode()
        
        st.sidebar.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 15px;">
                <img src="data:image/png;base64,{flag_data}" 
                     style="height: 40px; width: auto; border-radius: 6px;">
            </div>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass  # Silently continue if flag not found
    
    st.sidebar.title("Malawi Food Policy Simulator")
    PAGES = ["Overview", "Nutrition Analysis", "Policy Simulation", "Data Explorer"]
    if "page" not in st.session_state:
        st.session_state.page = PAGES[0]

    # Custom sidebar navigation (buttons)
    st.sidebar.markdown("#### Navigate to:")
    for idx, label in enumerate(PAGES):
        is_active = (st.session_state.page == label)
        btn_label = f"{label}"
        if st.sidebar.button(btn_label, key=f"nav_btn_{idx}"):
            st.session_state.page = label

    # Highlight active button via nth-of-type targeting
    active_index = PAGES.index(st.session_state.page) + 1
    st.markdown(
        f"""
        <style>
        section[data-testid="stSidebar"] .stButton > button {{
            width: 100%;
            justify-content: flex-start;
            background: rgba(255,255,255,0.08);
            color: #f0fff4;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }}
        section[data-testid="stSidebar"] .stButton:nth-of-type({active_index}) > button {{
            background: {PALETTE["primary_light"]} !important;
            color: {PALETTE["primary_darker"]} !important;
            border-color: {PALETTE["background_tint"]} !important;
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    page = st.session_state.page
    
    # Main content
    if page == "Overview":
        # Display Malawi flag
        display_flag()
        
        st.title("🌍 Malawi Food Security & Nutrition Dashboard")
        
        st.markdown("""
        This dashboard helps policymakers analyze food consumption patterns in Malawi and simulate interventions 
        to improve nutritional outcomes. Use the navigation panel to explore different aspects of food security.
        """)
        
        # Key metrics (unique colored cards)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_foods = len(malawi_consumption['Item'].unique()) if not malawi_consumption.empty else 0
            render_metric_card("Food Items Tracked", f"{total_foods}", background_color=PALETTE["primary_light"], text_color=PALETTE["primary_darker"])
        
        with col2:
            nutrients_count = len(nutrients) if not food_comp.empty else 0
            render_metric_card("Nutrients Analyzed", f"{nutrients_count}", background_color=PALETTE["background_tint"], text_color=PALETTE["primary_darker"])
        
        with col3:
            if not adequacy_df.empty:
                districts_count = len(adequacy_df['District'].unique())
                render_metric_card("Districts Analyzed", f"{districts_count}", background_color=PALETTE["primary_dark"], text_color="#f0fff4")
            else:
                render_metric_card("Districts Analyzed", "0", background_color=PALETTE["primary_dark"], text_color="#f0fff4")

        # Add vertical space to separate cards from next content
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        
        tabs = st.tabs(["Consumption", "Composition", "Highlights"])

        with tabs[0]:
            st.subheader("Top Food Consumption in Malawi")
            if not malawi_consumption.empty:
                top_foods = malawi_consumption.nlargest(10, 'Value')
                fig = px.bar(
                    top_foods, x='Item', y='Value',
                    title="Top 10 Food Items in Malawi (kg/capita/year)",
                    color_discrete_sequence=[PALETTE["primary_light"]]
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No consumption data available for Malawi")
        
        with tabs[1]:
            st.subheader("Nutritional Composition of Food Groups")
            if not food_comp.empty:
                food_groups = st.multiselect("Select Food Groups", 
                                            food_comp['Food_name'].unique(),
                                            default=food_comp['Food_name'].unique()[:3])
                if food_groups:
                    selected_foods = food_comp[food_comp['Food_name'].isin(food_groups)]
                    nutrient_select = st.selectbox("Select Nutrient", nutrients)
                    fig = px.bar(
                        selected_foods, x='Food_name', y=nutrient_select,
                        title=f"{nutrient_select} Content in Selected Foods",
                        color_discrete_sequence=[PALETTE["primary_light"]]
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No food composition data available")

        with tabs[2]:
            if not adequacy_df.empty:
                st.subheader("District vs Nutrient Heatmap")
                heat_fig = create_district_nutrient_heatmap(adequacy_df)
                st.plotly_chart(heat_fig, use_container_width=True)

    elif page == "Nutrition Analysis":
        st.title("📊 Malawi Nutrition Analysis")
        
        if not adequacy_df.empty:
            nutrient_cols = [col for col in adequacy_df.columns if col != 'District']

            tabs = st.tabs(["By District", "Deficiency", "Correlations", "Distributions"])
            
            with tabs[0]:
                st.subheader("Nutrient Adequacy by District")
                selected_nutrient = st.selectbox("Select Nutrient", nutrient_cols, key="na_bar")
                if selected_nutrient in adequacy_df.columns:
                    fig = px.bar(
                        adequacy_df, x='District', y=selected_nutrient,
                        title=f"{selected_nutrient} Adequacy by District in Malawi",
                        color_discrete_sequence=[PALETTE["primary_light"]]
                    )
                    fig.update_layout(xaxis_tickangle=-45, template=get_plotly_template())
                    st.plotly_chart(fig, use_container_width=True)
            
            with tabs[1]:
                st.subheader("Nutrient Deficiency Overview in Malawi")
                fig = create_deficiency_heatmap(adequacy_df)
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[2]:
                st.subheader("Nutrient Correlation Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    x_nutrient = st.selectbox("X-Axis Nutrient", nutrient_cols, key="x_nutrient")
                with col2:
                    y_nutrient = st.selectbox("Y-Axis Nutrient", nutrient_cols, key="y_nutrient")
                fig = create_interactive_scatter(adequacy_df, x_nutrient, y_nutrient)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[3]:
                st.subheader("Distribution of Nutrient Adequacy")
                dist_nutrient = st.selectbox("Select Nutrient for Distribution", nutrient_cols, key="dist_nutrient")
                col1, col2 = st.columns(2)
                with col1:
                    hist = px.histogram(
                        adequacy_df, x=dist_nutrient, nbins=20,
                        title=f"Histogram of {dist_nutrient} Adequacy",
                        color_discrete_sequence=[PALETTE["primary_light"]]
                    )
                    hist.update_layout(template=get_plotly_template())
                    st.plotly_chart(hist, use_container_width=True)
                with col2:
                    box = px.box(
                        adequacy_df, y=dist_nutrient,
                        title=f"Box Plot of {dist_nutrient} Adequacy",
                        color_discrete_sequence=[PALETTE["primary_light"]]
                    )
                    box.update_layout(template=get_plotly_template())
                    st.plotly_chart(box, use_container_width=True)
            
        else:
            st.warning("No nutrient adequacy data available")
        
        # Gender comparison if available
        if not gender_df.empty:
            st.subheader("Gender-based Nutrient Adequacy")
            
            # Select district and nutrient for gender comparison
            districts = gender_df['District'].unique()
            selected_district = st.selectbox("Select District", districts)
            
            # Get available nutrients from gender data
            gender_nutrients = [col for col in gender_df.columns if col not in ['District', 'Category']]
            
            if gender_nutrients:
                selected_gender_nutrient = st.selectbox("Select Nutrient for Gender Comparison", gender_nutrients)
                
                # Filter data for selected district and nutrient
                district_gender_data = gender_df[gender_df['District'] == selected_district]
                
                if not district_gender_data.empty:
                    # Create bar chart
                    fig = px.bar(district_gender_data, x='Category', y=selected_gender_nutrient, 
                                 color='Category', title=f"{selected_gender_nutrient} by Gender in {selected_district}")
                    st.plotly_chart(fig, use_container_width=True)

    elif page == "Policy Simulation":
        st.title("🎯 Malawi Policy Intervention Simulator")
        
        st.markdown("""
        Simulate the impact of different policy interventions on nutritional outcomes in Malawi.
        Select a district and intervention type to see potential effects.
        """)
        
        if not adequacy_df.empty:
            # Select district
            districts = adequacy_df['District'].unique()
            district = st.selectbox("Select District for Intervention", districts)
            
            # Intervention types - styled as buttons
            st.markdown("**Select Intervention Type**")
            intervention_options = ["Supplementation", "Fortification", "Diversification", "Subsidy"]
            
            if "intervention_type" not in st.session_state:
                st.session_state.intervention_type = intervention_options[0]
            
            cols = st.columns(4)
            for i, option in enumerate(intervention_options):
                with cols[i]:
                    is_active = st.session_state.intervention_type == option
                    if st.button(option, key=f"intervention_{i}"):
                        st.session_state.intervention_type = option
            
            intervention_type = st.session_state.intervention_type
            
            # Intervention parameters
            st.subheader("Intervention Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                nutrient_cols = [col for col in adequacy_df.columns if col != 'District']
                target_nutrient = st.selectbox("Target Nutrient", nutrient_cols)
                
            with col2:
                intensity = st.slider("Intervention Intensity", 1, 100, 10)
            
            # Simulate intervention effect
            if st.button("Run Simulation"):
                st.subheader("Simulation Results")
                
                # Get baseline data
                baseline_value = adequacy_df[adequacy_df['District'] == district][target_nutrient].values[0]
                
                # Calculate intervention effect based on type
                if intervention_type == "Supplementation":
                    effect = baseline_value * (intensity / 100)
                elif intervention_type == "Fortification":
                    effect = baseline_value * (intensity / 150)
                elif intervention_type == "Diversification":
                    effect = baseline_value * (intensity / 120)
                else:  # Subsidy
                    effect = baseline_value * (intensity / 80)
                
                projected_value = min(100, baseline_value + effect)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Current Adequacy", f"{baseline_value:.1f}%")
                col2.metric("Projected Adequacy", f"{projected_value:.1f}%")
                col3.metric("Improvement", f"+{effect:.1f}%", f"+{(effect/baseline_value*100):.1f}%" if baseline_value > 0 else "N/A")
                
                # Visualize results
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Current', 'Projected'],
                    y=[baseline_value, projected_value],
                    marker_color=[PALETTE["primary_darker"], PALETTE["primary_light"]]
                ))
                fig.update_layout(
                    title=f"Impact of {intervention_type} on {target_nutrient} in {district}",
                    yaxis_title=f"{target_nutrient} Adequacy (%)",
                    yaxis_range=[0, 100],
                    template=get_plotly_template()
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show pre-calculated simulation results if available
                if not simulations_df.empty:
                    st.subheader("Pre-calculated Intervention Impacts")
                    country_simulations = simulations_df[simulations_df['country'].str.lower() == 'malawi']
                    
                    if not country_simulations.empty:
                        # Check which columns exist in the simulations data
                        available_columns = country_simulations.columns.tolist()
                        
                        # Create visualizations based on available data
                        if 'adequacy_uplift_points' in available_columns:
                            fig1 = px.bar(country_simulations, x='intervention', y='adequacy_uplift_points', 
                                         color='intervention', title="Pre-calculated Intervention Impacts for Malawi",
                                         color_discrete_sequence=[PALETTE["primary_light"], PALETTE["primary_dark"]])
                            fig1.update_layout(template=get_plotly_template())
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        # Check if 'impact' column exists, otherwise use a different column
                        impact_column = 'impact' if 'impact' in available_columns else available_columns[1] if len(available_columns) > 1 else available_columns[0]
                        
                        if impact_column in available_columns:
                            fig2 = px.bar(country_simulations, x='intervention', y=impact_column, 
                                         color='intervention', title=f"Pre-calculated Intervention Impacts for {district}",
                                         color_discrete_sequence=[PALETTE["primary_light"], PALETTE["primary_dark"]])
                            fig2.update_layout(template=get_plotly_template())
                            st.plotly_chart(fig2, use_container_width=True)

                # Multi-scenario quick compare
                st.subheader("Compare Scenarios")
                scenarios = [
                    ("Low", 0.5),
                    ("Medium", 1.0),
                    ("High", 1.5)
                ]
                comp_vals = []
                for label, multiplier in scenarios:
                    if intervention_type == "Supplementation":
                        eff = baseline_value * (intensity / 100) * multiplier
                    elif intervention_type == "Fortification":
                        eff = baseline_value * (intensity / 150) * multiplier
                    elif intervention_type == "Diversification":
                        eff = baseline_value * (intensity / 120) * multiplier
                    else:
                        eff = baseline_value * (intensity / 80) * multiplier
                    comp_vals.append(min(100, baseline_value + eff))
                comp_fig = px.bar(
                    x=[s[0] for s in scenarios], y=comp_vals,
                    labels={'x': 'Scenario', 'y': f'{target_nutrient} Adequacy (%)'},
                    title="Scenario Comparison",
                    color_discrete_sequence=[PALETTE["primary_light"]]
                )
                comp_fig.update_layout(template=get_plotly_template())
                st.plotly_chart(comp_fig, use_container_width=True)
        else:
            st.warning("No nutrient adequacy data available for simulation")

    elif page == "Data Explorer":
        st.title("🔍 Malawi Data Explorer")
        
        st.markdown("Explore the underlying data used in this dashboard.")
        
        tabs = st.tabs(["Food Composition", "Consumption", "Adequacy", "Gender", "Simulations"])
        
        with tabs[0]:
            st.subheader("Food Composition Data")
            if not food_comp.empty:
                st.dataframe(food_comp, use_container_width=True)
            else:
                st.warning("No food composition data available")
                
        with tabs[1]:
            st.subheader("Consumption Patterns Data")
            if not malawi_consumption.empty:
                st.dataframe(malawi_consumption, use_container_width=True)
            else:
                st.warning("No consumption data available")
                
        with tabs[2]:
            st.subheader("Nutrient Adequacy Data")
            if not adequacy_df.empty:
                st.dataframe(adequacy_df, use_container_width=True)
            else:
                st.warning("No nutrient adequacy data available")
                
        with tabs[3]:
            st.subheader("Gender Comparison Data")
            if not gender_df.empty:
                st.dataframe(gender_df, use_container_width=True)
            else:
                st.warning("No gender comparison data available")
                
        with tabs[4]:
            st.subheader("Simulations Data")
            if not simulations_df.empty:
                st.dataframe(simulations_df, use_container_width=True)
            else:
                st.warning("No simulations data available")

    # Footer
    st.markdown("---")
    
    # Display flag in footer
    try:
        with open("data/malawi/flag.png", "rb") as f:
            flag_data = base64.b64encode(f.read()).decode()
        
        st.markdown(
            f"""
            <div style='text-align: center; margin-bottom: 10px;'>
                <img src="data:image/png;base64,{flag_data}" 
                     style="height: 30px; width: auto; border-radius: 4px; margin-right: 10px; vertical-align: middle;">
                <span style="vertical-align: middle; font-size: 14px; color: var(--text);">
                    Malawi Food Policy Simulator | For informed decision-making in nutrition security
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.markdown(
            """
            <div style='text-align: center'>
                <p>Malawi Food Policy Simulator | For informed decision-making in nutrition security</p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()