import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import base64

# --- Helper Function for Downloading Data ---
def download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# --- Database Connection ---
def get_elo_data(include_all_races):
    conn = sqlite3.connect('f1_elo.db')
    table_name = 'driver_elo_all' if include_all_races else 'driver_elo'
    query = f"""
        SELECT d.driverRef, r.year, r.round, r.date, de.elo, r.raceId
        FROM {table_name} de
        JOIN drivers d ON de.driverId = d.driverId
        JOIN races r ON de.raceId = r.raceId
        ORDER BY r.date, r.round
    """
    df = pd.read_sql_query(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df['elo'] = pd.to_numeric(df['elo'])
    conn.close()
    return df

# --- Callback Functions for Session State Updates ---
def update_race_selection():
    st.session_state.selected_race_label = st.session_state.race_select

def update_driver_selection():
     st.session_state.selected_drivers = st.session_state.driver_select

def update_start_date():
    st.session_state.start_date = st.session_state.start_date_input

def update_end_date():
    st.session_state.end_date = st.session_state.end_date_input

def update_include_all_races():
     st.session_state.include_all_races = st.session_state.include_all_races

# --- Streamlit App ---
def main():
    print("Running Streamlit app...")

    st.title('Formula 1 Driver Elo Ratings Over Time')

    # --- Sidebar: Options ---
    st.sidebar.header("Options")

    if 'include_all_races' not in st.session_state:
        st.session_state.include_all_races = False

    include_all_races = st.sidebar.checkbox(
        "Include all races (DNFs, Retirements, etc.)",
        value=st.session_state.include_all_races,
        key='include_all_races',
        on_change=update_include_all_races
    )


    # --- Date Range Selection ---
    st.sidebar.header("Date Range Selection")
    elo_df_all = get_elo_data(include_all_races)
    min_date = elo_df_all['date'].min().to_pydatetime()
    max_date = elo_df_all['date'].max().to_pydatetime()

    if 'start_date' not in st.session_state:
        st.session_state.start_date = min_date
    if 'end_date' not in st.session_state:
        st.session_state.end_date = max_date

    start_date = st.sidebar.date_input("Start Date", st.session_state.start_date, min_value=min_date, max_value=max_date, key='start_date_input', on_change=update_start_date)
    end_date = st.sidebar.date_input("End Date", st.session_state.end_date, min_value=min_date, max_value=max_date, key='end_date_input', on_change=update_end_date)

    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.max.time())

    elo_df = elo_df_all[(elo_df_all['date'] >= start_date) & (elo_df_all['date'] <= end_date)].copy()

    # --- Specific Race View ---
    st.sidebar.header("Specific Race View")
    races = elo_df_all[['raceId', 'year', 'round', 'date']].drop_duplicates().sort_values('date')
    races['race_label'] = races['year'].astype(str) + " Round " + races['round'].astype(str)
    race_labels = races['race_label'].tolist()

    if 'selected_race_label' not in st.session_state:
        st.session_state.selected_race_label = race_labels[0]

    selected_race_label = st.sidebar.selectbox(
        "Select Race",
        options=race_labels,
        key='race_select',
        index=race_labels.index(st.session_state.selected_race_label),
        on_change=update_race_selection
    )

    selected_race_id = races[races['race_label'] == selected_race_label]['raceId'].iloc[0]

    # --- Driver Selection (with session state) ---
    st.sidebar.header("Select Drivers")
    all_drivers = elo_df['driverRef'].sort_values().unique()

    default_drivers = []
    if 'max_verstappen' in all_drivers:
        default_drivers.append('max_verstappen')
    if 'perez' in all_drivers:
        default_drivers.append('perez')

    if 'selected_drivers' not in st.session_state:
        st.session_state.selected_drivers = default_drivers

    selected_drivers = st.sidebar.multiselect(
        'Choose up to 10 drivers',
        options=list(all_drivers),
        default=st.session_state.selected_drivers,
        key='driver_select',
        on_change=update_driver_selection
    )

    if len(selected_drivers) > 10:
        st.sidebar.warning("Please select a maximum of 10 drivers.")
        selected_drivers = selected_drivers[:10]
        st.session_state.selected_drivers = selected_drivers



    # --- Customizable Colors ---
    if 'driver_colors' not in st.session_state:
        st.session_state.driver_colors = {}
    for driver in selected_drivers:
        if driver not in st.session_state.driver_colors:
            st.session_state.driver_colors[driver] = px.colors.qualitative.Plotly[selected_drivers.index(driver) % len(px.colors.qualitative.Plotly)]
        st.session_state.driver_colors[driver] = st.sidebar.color_picker(f"Color for {driver}", st.session_state.driver_colors[driver], key=f'color_{driver}')

    # --- Main Panel: Chart ---
    if selected_drivers:
        filtered_df = elo_df[elo_df['driverRef'].isin(selected_drivers)].copy()
        dates = filtered_df['date'].dt.to_pydatetime().tolist()
        elos = filtered_df['elo'].apply(lambda x: float(x) if pd.notna(x) else None).tolist()
        drivers = filtered_df['driverRef'].tolist()
        years = filtered_df['year'].tolist()
        rounds = filtered_df['round'].tolist()

        fig = go.Figure()
        for driver in selected_drivers:
            driver_filter = [d == driver for d in drivers]
            driver_dates = [d for d, flag in zip(dates, driver_filter) if flag]
            driver_elos = [elo for elo, flag in zip(elos, driver_filter) if flag]
            driver_years = [yr for yr, flag in zip(years, driver_filter) if flag]
            driver_rounds = [rnd for rnd, flag in zip(rounds, driver_filter) if flag]

            hover_text = [
                f"Driver: {driver}<br>Date: {date.strftime('%Y-%m-%d')}<br>Year: {yr}<br>Round: {rnd}<br>Elo: {elo:.2f}"
                for date, yr, rnd, elo in zip(driver_dates, driver_years, driver_rounds, driver_elos)
            ]

            fig.add_trace(go.Scatter(
                x=driver_dates,
                y=driver_elos,
                mode='lines',
                name=driver,
                hovertext=hover_text,
                hoverinfo="text",
                line=dict(color=st.session_state.driver_colors[driver])
            ))

        title = 'Driver Elo Rating Over Time'
        if include_all_races:
            title += ' (Including DNFs, Retirements, etc.)'
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Elo Rating',
            legend_title='Driver',
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("Select drivers in the sidebar to view their Elo ratings.")

    # --- Specific Race View: Elo Table ---
    st.header(f"Elo Ratings Before {selected_race_label}")
    race_data = elo_df_all[elo_df_all['raceId'] == selected_race_id]

    if not race_data.empty:
        drivers_in_race = race_data['driverRef'].unique()
        previous_race_data = elo_df_all[elo_df_all['date'] < race_data['date'].iloc[0]]
        previous_race_data = previous_race_data[previous_race_data['driverRef'].isin(drivers_in_race)]
        previous_race_data = previous_race_data.groupby('driverRef')['elo'].last().reset_index()
        previous_race_data = previous_race_data.sort_values('elo', ascending=False)

        if not previous_race_data.empty:
            current_race_elo = race_data.set_index('driverRef')['elo']
            previous_race_data = previous_race_data.set_index('driverRef')
            merged_data = pd.concat([previous_race_data, current_race_elo], axis=1, keys=['Elo Before', 'Elo After'])
            merged_data['Elo Change'] = merged_data['Elo After'] - merged_data['Elo Before']
            merged_data = merged_data.reset_index()
            st.dataframe(merged_data, use_container_width=True) # Use container width
        else:
            st.write("No previous race data available.")
    else:
        st.write("No data available for this race.")

    # --- Head-to-Head Comparison ---
    st.header("Head-to-Head Driver Comparison")
    if len(selected_drivers) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            driver1 = st.selectbox("Select Driver 1", options=selected_drivers, key="driver1_select")
        with col2:
            driver2 = st.selectbox("Select Driver 2", options=selected_drivers, key="driver2_select", index=1 if len(selected_drivers) > 1 else 0)

        if driver1 != driver2:
            driver1_data = elo_df[elo_df['driverRef'] == driver1]
            driver2_data = elo_df[elo_df['driverRef'] == driver2]

            merged_data = pd.merge(driver1_data, driver2_data, on='date', suffixes=(f'_{driver1}', f'_{driver2}'), how='outer')
            merged_data = merged_data.sort_values('date').fillna(method='ffill')

            dates = merged_data['date'].dt.to_pydatetime().tolist()
            elo1 = merged_data[f'elo_{driver1}'].apply(lambda x: float(x) if pd.notna(x) else None).tolist()
            elo2 = merged_data[f'elo_{driver2}'].apply(lambda x: float(x) if pd.notna(x) else None).tolist()
            elo_diff = [(e1 - e2) if (e1 is not None and e2 is not None) else None for e1, e2 in zip(elo1, elo2)]

            fig_diff = go.Figure()
            fig_diff.add_trace(go.Scatter(x=dates, y=elo_diff, mode='lines', name='Elo Difference'))
            fig_diff.update_layout(title=f'Elo Difference: {driver1} vs {driver2}', xaxis_title='Date', yaxis_title='Elo Difference')
            st.plotly_chart(fig_diff)
        else:
            st.write("Please select two different drivers for comparison.")
    else:
        st.write("Select at least two drivers for head-to-head comparison.")

    # --- Highest Elo Achieved ---
    st.header("Highest Elo Achieved (Within Date Range)")

    highest_elo_df = elo_df.loc[elo_df.groupby('driverRef')['elo'].idxmax()]
    highest_elo_df = highest_elo_df.sort_values(by='elo', ascending=False)
    highest_elo_df = highest_elo_df[['driverRef', 'elo', 'date', 'year', 'round']]
    highest_elo_df.rename(columns={'driverRef' : 'Driver',
                                   'elo': 'Highest Elo',
                                   'date': "Date Achieved",
                                    'year': "Year",
                                    'round': "Round"}, inplace=True)


    st.dataframe(highest_elo_df, use_container_width=True) # Use container width

    # --- Data Table ---
    st.header("Raw Data")
    if not elo_df.empty:
        search_term = st.text_input("Search", key="search_input")
        filtered_results = elo_df[elo_df.apply(lambda row: any(search_term.lower() in str(value).lower() for value in row), axis=1)] if search_term else elo_df
        st.dataframe(filtered_results, use_container_width=True)  # Use container width!
        st.markdown(download_link(filtered_results, "f1_elo_data.csv", "Download Data"), unsafe_allow_html=True)
    else:
        st.write("No data available for the selected date range.")

    # --- Elo Calculation Explanation ---
    st.header("How Elo Ratings are Calculated")
    with st.expander("See Explanation"):
        st.markdown("""
        The Elo rating system calculates relative skill levels.  Here's how it works in this F1 app:

        *   **Initial Rating:** Each driver and constructor begins with an initial Elo rating of 1000.
        *   **Expected vs. Actual:** Before each race, the system calculates an *expected* outcome for each driver and team, based on their current Elo ratings.  After the race, this is compared to the *actual* outcome.
        *   **Rating Updates:** Ratings are adjusted based on the difference between expected and actual results.  Outperforming expectations leads to a rating increase; underperforming leads to a decrease.
        *   **K-Factor:**  The K-factor determines the *magnitude* of rating changes.  A higher K-factor means larger, more volatile updates.
        *   **Constructor (Team) Elo:**  Each team *also* has an Elo rating, reflecting the overall performance of the car and team operations.
        *   **Team Influence:** A driver's Elo change is influenced by their team's Elo.  A driver in a strong car (high team Elo) is expected to perform well.  Their rating changes are more sensitive to their performance *relative to their teammate*.

        **Key Concepts:**

        *   **Zero-Sum:** Elo point gains by some are balanced by losses by others.
        *   **Relative Skill:** Elo measures relative skill within the group (drivers and teams in F1).
        *   **Predictive Power:** Elo differences predict the probability of one driver/team beating another.
        * **Team adjusted individual performance:** The driver's elo is not only about finishing position, but about relative performance compared to the teammate, considering the strength of the team.
        """)

if __name__ == '__main__':
    main()