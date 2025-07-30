import streamlit as st
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# Caching the data load to avoid re-fetching on every run
@st.cache_data(ttl=3600)
def load_census_data(api_key):
    BASE_URL = 'https://api.census.gov/data/2021/acs/acs5'
    variables = {
        "B19013_001E": "median_household_income",
        "B25035_001E": "median_year_built",
        "B01003_001E": "total_population",
        "B25003_001E": "total_occupied_units",
        "B25003_002E": "owner_occupied_units",
        "B25024_001E": "total_housing_units",
        "B25024_002E": "sf_detached",
        "B25024_003E": "sf_attached"
    }
    var_list = ",".join(variables.keys())

    response = requests.get(
        f"{BASE_URL}?get=NAME,{var_list}&for=county:*&key={api_key}"
    )
    response.raise_for_status()
    data = response.json()

    columns = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=columns)

    df = df.rename(columns=variables)
    for col in variables.values():
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df["fips"] = df["state"] + df["county"]
    df["homeownership_rate"] = df["owner_occupied_units"] / df["total_occupied_units"]
    df["sf_ratio"] = (df["sf_detached"] + df["sf_attached"]) / df["total_housing_units"].replace(0, pd.NA)

    return df

def id_to_fips(id_str):
    state_abbr_to_fips = {
        'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08', 'CT': '09',
        'DE': '10', 'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
        'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24', 'MA': '25',
        'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32',
        'NH': '33', 'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
        'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47',
        'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55',
        'WY': '56'
    }
    state_abbr, county_code = id_str.split("-")
    return state_abbr_to_fips[state_abbr] + county_code.zfill(3)

@st.cache_data(ttl=3600)
def load_weather_data():

    precip = pd.read_csv("avg_prec.csv")
    temp = pd.read_csv("avg_temp.csv")

    precip["fips"] = precip["ID"].apply(id_to_fips)
    temp["fips"] = temp["ID"].apply(id_to_fips)

    precip = precip[["fips", "Value"]].rename(columns={"Value": "avg_precipitation"})
    temp = temp[["fips", "Value"]].rename(columns={"Value": "avg_temperature"})

    return precip, temp

def calculate_scores(df):
    # Fill missing weather values with median
    # df['avg_precipitation'] = df['avg_precipitation'].fillna(df['avg_precipitation'].median())
    # df['avg_temperature'] = df['avg_temperature'].fillna(df['avg_temperature'].median())

    df["home_age"] = 2025 - df["median_year_built"]
    df["adj_population"] = df["total_population"] * df["sf_ratio"]

    df["rank_income"] = df["median_household_income"].rank(pct=True)
    df["rank_home_age"] = df["home_age"].rank(pct=True)
    df["rank_population"] = df["total_population"].rank(pct=True)
    df["rank_adj_population"] = df["adj_population"].rank(pct=True)
    df["rank_homeownership"] = df["homeownership_rate"].rank(pct=True)
    df['rank_avg_precipitation'] = df['avg_precipitation'].rank(pct=True)
    df['rank_avg_temperature'] = df['avg_temperature'].rank(pct=True)
    df["rank_sf_ratio"] = df["sf_ratio"].rank(pct=True)

    df["pest_sales_score_raw"] = (
        df["rank_home_age"] * 0.125 +
        df["rank_avg_precipitation"] * 0.125 +
        df["rank_avg_temperature"] * 0.125 +
        df["rank_sf_ratio"] * 0.25 +
        df["rank_income"] * 0.125 +
        df["rank_adj_population"] * 0.50 +
        df["rank_homeownership"] * 0.05
    )

    scaler = MinMaxScaler()
    df["pest_sales_score"] = scaler.fit_transform(df[["pest_sales_score_raw"]])

    return df

def main():
    st.title("Pest Sales Potential Dashboard")
    st.markdown("""
    This interactive dashboard predicts the potential for door-to-door pest control sales across U.S. counties.

    It combines demographic data, housing characteristics, and weather factors to highlight the best sales areas.

    Use the filters below to explore by state or zoom into specific regions.
    """)

    # Load data
    API_KEY = st.secrets["CENSUS_API_KEY"]

    if not API_KEY:
        st.warning("Please enter a Census API key to proceed.")
        return

    with st.spinner("Loading Census data..."):
        df = load_census_data(API_KEY)

    with st.spinner("Loading weather data..."):
        precip, temp = load_weather_data()

    df = df.merge(precip, on="fips", how="left")
    df = df.merge(temp, on="fips", how="left")

    df = calculate_scores(df)


    # Filter by state
    state_fips_to_name = {
        '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas', '06': 'California',
        '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware', '12': 'Florida', '13': 'Georgia',
        '15': 'Hawaii', '16': 'Idaho', '17': 'Illinois', '18': 'Indiana', '19': 'Iowa',
        '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine', '24': 'Maryland',
        '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota', '28': 'Mississippi', '29': 'Missouri',
        '30': 'Montana', '31': 'Nebraska', '32': 'Nevada', '33': 'New Hampshire', '34': 'New Jersey',
        '35': 'New Mexico', '36': 'New York', '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio',
        '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island', '45': 'South Carolina',
        '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas', '49': 'Utah', '50': 'Vermont',
        '51': 'Virginia', '53': 'Washington', '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming'
    }

    # Create a new column for full state names in df:
    df['state_name'] = df['state'].map(state_fips_to_name)

    # Add an "All States" option
    state_options = ['All States'] + sorted(df['state_name'].dropna().unique().tolist())

    selected_state = st.selectbox("Filter by State", options=state_options, index=0)

    if selected_state != 'All States':
        filtered_df = df[df['state_name'] == selected_state]
    else:
        filtered_df = df


    st.subheader(f"Sales Potential for {selected_state}")

    st.dataframe(filtered_df[['NAME', 'median_household_income', 'home_age', 'sf_ratio', 'pest_sales_score']].sort_values('pest_sales_score', ascending=False).head(10))

    # Plot map
    geojson_url = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'

    fig = px.choropleth(
        filtered_df,
        geojson=geojson_url,
        locations='fips',
        color='pest_sales_score',
        color_continuous_scale='RdYlGn',
        scope='usa',
        labels={'pest_sales_score': 'Sales Potential'},
        hover_name='NAME',
        hover_data={
            'median_household_income':':$,.0f',
            'pest_sales_score':':.2f',
            'sf_ratio':':.2f',
            'home_age':':.0f'
        },
        title=f'Predicted Pest Sales Potential by County in {selected_state}'
    )

    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        <small>
        This dashboard predicts the potential for door-to-door pest control sales across U.S. counties by combining key factors:
        <ul>
        <li><b>Median Household Income</b>: Wealthier areas may afford pest control more readily.</li>
        <li><b>Median Home Age</b>: Older homes might have more pest issues.</li>
        <li><b>Total Population</b>: More people means more potential customers.</li>
        <li><b>Homeownership Rate</b>: Owners may be more likely to invest in pest control than renters.</li>
        <li><b>Single Family Home Ratio</b>: Focus on areas with more detached single-family homes (vs. apartments).</li>
        <li><b>Average Precipitation & Temperature</b>: Weather conditions influence pest activity.</li>
        </ul>
        </small>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
