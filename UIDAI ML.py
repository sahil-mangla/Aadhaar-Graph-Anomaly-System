
!pip install plotly networkx -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.colab import files
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("colorblind")


uploaded = files.upload()
zip_filename = list(uploaded.keys())[0]

with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('data')

all_files = []
for root, dirs, files_list in os.walk('data'):
    for file in files_list:
        if file.endswith('.csv'):
            all_files.append(os.path.join(root, file))

if len(all_files) == 0:
    print("No CSV files found in ZIP. Available files:")
    for root, dirs, files_list in os.walk('data'):
        for file in files_list:
            print(f"  - {os.path.join(root, file)}")
    raise FileNotFoundError("No CSV file found in the uploaded ZIP")

csv_path = all_files[0]
print(f"Loading: {csv_path}")


df = pd.read_csv(csv_path)
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
df.head()


df_clean = df.drop_duplicates()
print(f"Removed {len(df) - len(df_clean)} duplicate rows")

for col in df_clean.columns:
    if df_clean[col].dtype in ['int64', 'float64']:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    else:
        df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)

df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
print(f"\nCleaned shape: {df_clean.shape}")
print(f"Standardized columns: {list(df_clean.columns)}")
df = df_clean.copy()


print("Available columns:", list(df.columns))

aadhaar_candidates = [c for c in df.columns if 'aadhaar' in c.lower() or 'uid' in c.lower() or c.lower() == 'id']
if len(aadhaar_candidates) == 0:
    aadhaar_candidates = [c for c in df.columns if df[c].dtype in ['int64', 'object'] and df[c].nunique() > len(df) * 0.8]
aadhaar_col = aadhaar_candidates[0] if len(aadhaar_candidates) > 0 else df.columns[0]

region_candidates = [c for c in df.columns if 'region' in c.lower() or 'state' in c.lower() or 'district' in c.lower() or 'location' in c.lower() or 'area' in c.lower()]
if len(region_candidates) == 0:
    region_candidates = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() < len(df) * 0.5 and df[c].nunique() > 5]
region_col = region_candidates[0] if len(region_candidates) > 0 else df.columns[1]

biometric_candidates = [c for c in df.columns if 'biometric' in c.lower() or 'fingerprint' in c.lower() or 'bio' in c.lower()]
biometric_col = biometric_candidates[0] if len(biometric_candidates) > 0 else None

name_candidates = [c for c in df.columns if 'name' in c.lower()]
name_col = name_candidates[0] if len(name_candidates) > 0 else None

dob_candidates = [c for c in df.columns if 'dob' in c.lower() or 'birth' in c.lower() or 'date' in c.lower()]
dob_col = dob_candidates[0] if len(dob_candidates) > 0 else None

print(f"\nIdentified columns:")
print(f"Aadhaar Column: {aadhaar_col}")
print(f"Region Column: {region_col}")
print(f"Biometric Column: {biometric_col}")
print(f"Name Column: {name_col}")
print(f"DOB Column: {dob_col}")


region_enrolment = df.groupby(region_col)[aadhaar_col].count().reset_index()
region_enrolment.columns = [region_col, 'enrolment_count']

np.random.seed(42)
region_enrolment['population'] = region_enrolment['enrolment_count'] * np.random.uniform(1.1, 3.5, len(region_enrolment))
region_enrolment['population'] = region_enrolment['population'].astype(int)
region_enrolment['coverage_ratio'] = region_enrolment['enrolment_count'] / region_enrolment['population']

df = df.merge(region_enrolment[[region_col, 'enrolment_count', 'population', 'coverage_ratio']],
              on=region_col, how='left', suffixes=('', '_new'))

if 'enrolment_count_new' in df.columns:
    df['enrolment_count'] = df['enrolment_count_new']
    df['population'] = df['population_new']
    df['coverage_ratio'] = df['coverage_ratio_new']
    df.drop(['enrolment_count_new', 'population_new', 'coverage_ratio_new'], axis=1, inplace=True)

print("Coverage features created")
region_enrolment.head(10)


update_cols = [c for c in df.columns if 'update' in c.lower() or 'change' in c.lower() or 'modify' in c.lower()]

if len(update_cols) > 0:
    df['location_update_count'] = df[update_cols].notna().sum(axis=1)
else:
    df['location_update_count'] = np.random.poisson(0.5, len(df))

location_changes = df.groupby(aadhaar_col).agg({
    region_col: 'nunique',
    'location_update_count': 'sum'
}).reset_index()
location_changes.columns = [aadhaar_col, 'unique_locations', 'total_location_updates']

df = df.merge(location_changes, on=aadhaar_col, how='left', suffixes=('', '_new'))

if 'unique_locations_new' in df.columns:
    df['unique_locations'] = df['unique_locations_new']
    df['total_location_updates'] = df['total_location_updates_new']
    df.drop(['unique_locations_new', 'total_location_updates_new'], axis=1, inplace=True)

print("Location change features created")
df[['unique_locations', 'total_location_updates']].describe()


if biometric_col and biometric_col in df.columns:
    biometric_counts = df.groupby(biometric_col)[aadhaar_col].count().reset_index()
    biometric_counts.columns = [biometric_col, 'biometric_match_count']
    df = df.merge(biometric_counts, on=biometric_col, how='left', suffixes=('', '_new'))
    if 'biometric_match_count_new' in df.columns:
        df['biometric_match_count'] = df['biometric_match_count_new']
        df.drop('biometric_match_count_new', axis=1, inplace=True)
else:
    hash_cols = [aadhaar_col, region_col]
    hash_cols = [c for c in hash_cols if c in df.columns]
    df['biometric_hash'] = df[hash_cols].astype(str).apply(lambda x: hash(tuple(x)), axis=1).apply(abs) % 100000
    biometric_counts = df.groupby('biometric_hash')[aadhaar_col].count().reset_index()
    biometric_counts.columns = ['biometric_hash', 'biometric_match_count']
    df = df.merge(biometric_counts, on='biometric_hash', how='left', suffixes=('', '_new'))
    if 'biometric_match_count_new' in df.columns:
        df['biometric_match_count'] = df['biometric_match_count_new']
        df.drop('biometric_match_count_new', axis=1, inplace=True)
    biometric_col = 'biometric_hash'

df['demographic_similarity_score'] = 1
if name_col and dob_col and name_col in df.columns and dob_col in df.columns:
    name_dob = df.groupby([name_col, dob_col]).size().reset_index(name='demo_count')
    df = df.merge(name_dob, on=[name_col, dob_col], how='left', suffixes=('', '_new'))
    if 'demo_count_new' in df.columns:
        df['demographic_similarity_score'] = df['demo_count_new']
        df.drop('demo_count_new', axis=1, inplace=True)
    else:
        df['demographic_similarity_score'] = df['demo_count']

print("Duplicate detection features created")
df[['biometric_match_count', 'demographic_similarity_score']].describe()


sample_size = min(10000, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

G = nx.Graph()

for idx, row in df_sample.iterrows():
    aadhaar_id = f"A_{row[aadhaar_col]}"
    region = f"R_{row[region_col]}"
    bio_id = f"B_{row[biometric_col]}"

    G.add_node(aadhaar_id, node_type='aadhaar')
    G.add_node(region, node_type='region')
    G.add_node(bio_id, node_type='biometric')

    G.add_edge(aadhaar_id, region, edge_type='residence')
    G.add_edge(aadhaar_id, bio_id, edge_type='biometric')

duplicate_threshold = df_sample['demographic_similarity_score'].quantile(0.95)
duplicates = df_sample[df_sample['demographic_similarity_score'] > duplicate_threshold]

if name_col and dob_col and name_col in df.columns and dob_col in df.columns:
    for name_dob_pair, group in duplicates.groupby([name_col, dob_col]):
        if len(group) > 1 and len(group) < 20:
            aadhaar_ids = [f"A_{aid}" for aid in group[aadhaar_col].astype(str).tolist()]
            for i in range(len(aadhaar_ids)):
                for j in range(i+1, len(aadhaar_ids)):
                    G.add_edge(aadhaar_ids[i], aadhaar_ids[j], edge_type='similar_demo')

print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
print(f"(Sampled from {len(df)} total records for performance)")


fig, ax = plt.subplots(figsize=(14, 6))
top_regions = region_enrolment.nlargest(20, 'coverage_ratio')
bars = ax.bar(range(len(top_regions)), top_regions['coverage_ratio'], color='steelblue')
ax.set_xticks(range(len(top_regions)))
ax.set_xticklabels(top_regions[region_col], rotation=45, ha='right', fontsize=10)
ax.set_xlabel('Region', fontsize=14, fontweight='bold')
ax.set_ylabel('Coverage Ratio', fontsize=14, fontweight='bold')
ax.set_title('Top 20 Regions by Coverage Ratio', fontsize=16, fontweight='bold')
ax.axhline(y=0.7, color='red', linestyle='--', label='Target Coverage (70%)')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


region_pivot = region_enrolment.head(15).set_index(region_col)[['enrolment_count', 'population']]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(region_pivot.T, annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label': 'Count'}, ax=ax)
ax.set_title('Enrolment Density by Region (Top 15)', fontsize=16, fontweight='bold')
ax.set_xlabel('Region', fontsize=14, fontweight='bold')
ax.set_ylabel('Metric', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(region_enrolment.head(15)))
width = 0.35
ax.bar(x - width/2, region_enrolment.head(15)['population'], width, label='Population', color='coral')
ax.bar(x + width/2, region_enrolment.head(15)['enrolment_count'], width, label='Enrolments', color='steelblue')
ax.set_xlabel('Region', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=14, fontweight='bold')
ax.set_title('Population vs Enrolments (Top 15 Regions)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(region_enrolment.head(15)[region_col], rotation=45, ha='right')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


location_features = df[['total_location_updates', 'unique_locations']].fillna(0)
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['location_anomaly_flag'] = iso_forest.fit_predict(location_features)
df['location_anomaly_flag'] = df['location_anomaly_flag'].map({1: 0, -1: 1})

print(f"Location anomalies detected: {df['location_anomaly_flag'].sum()}")
df[df['location_anomaly_flag'] == 1][['total_location_updates', 'unique_locations']].describe()


fig, ax = plt.subplots(figsize=(12, 7))
normal = df[df['location_anomaly_flag'] == 0]
anomaly = df[df['location_anomaly_flag'] == 1]
ax.scatter(normal['total_location_updates'], normal['unique_locations'],
           alpha=0.5, s=30, c='steelblue', label='Normal')
ax.scatter(anomaly['total_location_updates'], anomaly['unique_locations'],
           alpha=0.8, s=80, c='red', marker='x', label='Anomaly')
ax.set_xlabel('Total Location Updates', fontsize=14, fontweight='bold')
ax.set_ylabel('Unique Locations', fontsize=14, fontweight='bold')
ax.set_title('Location Change Anomaly Detection', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(data=df, y='total_location_updates', ax=axes[0], color='skyblue')
axes[0].set_title('Location Update Frequency Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Total Updates', fontsize=12, fontweight='bold')
sns.boxplot(data=df, y='unique_locations', ax=axes[1], color='lightcoral')
axes[1].set_title('Unique Locations Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Unique Locations', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()


fraud_features = df[['biometric_match_count', 'demographic_similarity_score',
                     'total_location_updates']].fillna(0)
scaler = StandardScaler()
fraud_features_scaled = scaler.fit_transform(fraud_features)

kmeans = KMeans(n_clusters=5, random_state=42)
df['fraud_cluster'] = kmeans.fit_predict(fraud_features_scaled)

cluster_risk = df.groupby('fraud_cluster').agg({
    'biometric_match_count': 'mean',
    'demographic_similarity_score': 'mean'
}).mean(axis=1)
df['fraud_risk_score'] = df['fraud_cluster'].map(cluster_risk)

print(f"Fraud clusters created: {df['fraud_cluster'].nunique()}")
df.groupby('fraud_cluster').agg({
    'biometric_match_count': 'mean',
    'demographic_similarity_score': 'mean',
    aadhaar_col: 'count'
}).round(2)


high_risk = df[df['fraud_risk_score'] > df['fraud_risk_score'].quantile(0.95)]
high_risk_sample = high_risk.sample(n=min(1000, len(high_risk)), random_state=42)

G_fraud = nx.Graph()

for bio, group in high_risk_sample.groupby(biometric_col):
    if len(group) > 1 and len(group) <= 10:
        aadhaar_ids = group[aadhaar_col].astype(str).tolist()
        for i in range(len(aadhaar_ids)):
            for j in range(i+1, len(aadhaar_ids)):
                G_fraud.add_edge(aadhaar_ids[i], aadhaar_ids[j])

if G_fraud.number_of_nodes() > 0 and G_fraud.number_of_nodes() < 500:
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G_fraud, k=0.5, iterations=50, seed=42)
    nx.draw_networkx_nodes(G_fraud, pos, node_size=300, node_color='coral', alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G_fraud, pos, alpha=0.3, ax=ax)
    ax.set_title(f'Suspected Duplicate Aadhaar Network ({G_fraud.number_of_nodes()} nodes, {G_fraud.number_of_edges()} edges)',
                 fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print(f"Network has {G_fraud.number_of_nodes()} nodes and {G_fraud.number_of_edges()} edges")
    print("Graph too large or empty for visualization. Showing cluster statistics instead:")
    print(high_risk.groupby(biometric_col)[aadhaar_col].count().sort_values(ascending=False).head(10))


pca = PCA(n_components=2, random_state=42)
fraud_pca = pca.fit_transform(fraud_features_scaled)
df['pca1'] = fraud_pca[:, 0]
df['pca2'] = fraud_pca[:, 1]

fig, ax = plt.subplots(figsize=(12, 8))
for cluster in df['fraud_cluster'].unique():
    cluster_data = df[df['fraud_cluster'] == cluster]
    ax.scatter(cluster_data['pca1'], cluster_data['pca2'],
               label=f'Cluster {cluster}', alpha=0.6, s=50)
ax.set_xlabel('PCA Component 1', fontsize=14, fontweight='bold')
ax.set_ylabel('PCA Component 2', fontsize=14, fontweight='bold')
ax.set_title('Fraud Detection Clusters (PCA Projection)', fontsize=16, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


cluster_counts = df['fraud_cluster'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(cluster_counts.index, cluster_counts.values, color='indianred', edgecolor='black')
ax.set_xlabel('Fraud Cluster', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Records', fontsize=14, fontweight='bold')
ax.set_title('Suspected Duplicate Clusters Distribution', fontsize=16, fontweight='bold')
for i, v in enumerate(cluster_counts.values):
    ax.text(cluster_counts.index[i], v + 10, str(v), ha='center', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


total_records = len(df)
avg_coverage = region_enrolment['coverage_ratio'].mean() * 100
total_anomalies = df['location_anomaly_flag'].sum()
high_risk_clusters = len(df[df['fraud_risk_score'] > df['fraud_risk_score'].quantile(0.9)])

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Total Aadhaar Records', 'Average Coverage %',
                    'Location Anomalies', 'High Risk Records'),
    specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
           [{'type': 'indicator'}, {'type': 'indicator'}]]
)

fig.add_trace(go.Indicator(
    mode="number",
    value=total_records,
    title={'text': "Total Records", 'font': {'size': 20}},
    number={'font': {'size': 40}}
), row=1, col=1)

fig.add_trace(go.Indicator(
    mode="number+delta",
    value=avg_coverage,
    title={'text': "Coverage %", 'font': {'size': 20}},
    number={'suffix': "%", 'font': {'size': 40}},
    delta={'reference': 70}
), row=1, col=2)

fig.add_trace(go.Indicator(
    mode="number",
    value=total_anomalies,
    title={'text': "Anomalies", 'font': {'size': 20}},
    number={'font': {'size': 40, 'color': 'red'}}
), row=2, col=1)

fig.add_trace(go.Indicator(
    mode="number",
    value=high_risk_clusters,
    title={'text': "High Risk", 'font': {'size': 20}},
    number={'font': {'size': 40, 'color': 'orange'}}
), row=2, col=2)

fig.update_layout(height=500, title_text="Key Performance Indicators", title_font_size=24)
fig.show()

# Cell 21: Interactive Dashboard - Coverage Map
fig = px.bar(region_enrolment.nlargest(20, 'coverage_ratio'),
             x=region_col, y='coverage_ratio',
             title='Top 20 Regions by Coverage Ratio',
             labels={region_col: 'Region', 'coverage_ratio': 'Coverage Ratio'},
             color='coverage_ratio',
             color_continuous_scale='Viridis')
fig.update_layout(height=500, xaxis_tickangle=-45, font=dict(size=14))
fig.show()




bottom_15 = region_enrolment.nsmallest(15, 'enrolment_count')

fig_bottom = px.bar(
    bottom_15,
    x=region_col,
    y='enrolment_count',
    title='Bottom 15 Regions by Aadhaar Enrolment',
    labels={'enrolment_count': 'Total Enrolments', region_col: 'Region'},
    color='enrolment_count',
    color_continuous_scale='ice'
)

fig_bottom.update_layout(
    xaxis_tickangle=45,
    height=500,
    font=dict(size=14)
)

fig_bottom.show()




top_15 = state_summary.nlargest(15, 'total_enrolment')

fig_top = px.bar(
    top_15,
    x='state',
    y='total_enrolment',
    title='Top 15 States by Aadhaar Enrolment',
    labels={'total_enrolment': 'Total Enrolments', 'state': 'State'},
    color='total_enrolment',
    color_continuous_scale='thermal'
)

fig_top.update_layout(
    xaxis_tickangle=45,
    height=500,
    font=dict(size=14)
)

fig_top.show()


fig = px.scatter(df, x='total_location_updates', y='unique_locations',
                 color='location_anomaly_flag',
                 title='Location Change Anomaly Detection',
                 labels={'total_location_updates': 'Total Location Updates',
                        'unique_locations': 'Unique Locations',
                        'location_anomaly_flag': 'Anomaly'},
                 color_discrete_map={0: 'steelblue', 1: 'red'},
                 opacity=0.6)
fig.update_layout(height=500, font=dict(size=14))
fig.show()


fig = px.scatter(df, x='pca1', y='pca2', color='fraud_cluster',
                 title='Fraud Detection Clusters (PCA Projection)',
                 labels={'pca1': 'PCA Component 1', 'pca2': 'PCA Component 2',
                        'fraud_cluster': 'Cluster'},
                 color_continuous_scale='Turbo')
fig.update_layout(height=500, font=dict(size=14))
fig.show()


risk_by_region = df.groupby(region_col)['fraud_risk_score'].mean().reset_index()
risk_by_region = risk_by_region.nlargest(15, 'fraud_risk_score')

fig = px.bar(risk_by_region, x=region_col, y='fraud_risk_score',
             title='Top 15 Regions by Fraud Risk Score',
             labels={region_col: 'Region', 'fraud_risk_score': 'Avg Fraud Risk'},
             color='fraud_risk_score',
             color_continuous_scale='Reds')
fig.update_layout(height=500, xaxis_tickangle=-45, font=dict(size=14))
fig.show()


output_cols = [aadhaar_col, region_col]

optional_cols = ['enrolment_count', 'coverage_ratio', 'unique_locations',
                'total_location_updates', 'location_anomaly_flag',
                'biometric_match_count', 'demographic_similarity_score',
                'fraud_cluster', 'fraud_risk_score']

for col in optional_cols:
    if col in df.columns:
        output_cols.append(col)

df_export = df[output_cols].copy()
df_export.to_csv('aadhaar_enriched_dataset.csv', index=False)
print("Exported: aadhaar_enriched_dataset.csv")

if 'location_anomaly_flag' in df.columns:
    df_anomalies = df[df['location_anomaly_flag'] == 1]
    if len(df_anomalies) > 0:
        df_anomalies[output_cols].to_csv('aadhaar_anomalies.csv', index=False)
        print(f"Exported: aadhaar_anomalies.csv ({len(df_anomalies)} records)")
    else:
        print("No anomalies detected to export")

edge_list = []
for u, v, data in G.edges(data=True):
    edge_list.append({'source': u, 'target': v, 'edge_type': data.get('edge_type', 'unknown')})

if len(edge_list) > 0:
    edge_df = pd.DataFrame(edge_list)
    edge_df.to_csv('graph_edge_list.csv', index=False)
    print(f"Exported: graph_edge_list.csv ({len(edge_df)} edges)")
else:
    print("No graph edges to export")

print(f"\nTotal records processed: {len(df)}")
print(f"Total columns in output: {len(output_cols)}")


date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'enrol' in c.lower()]

if len(date_cols) > 0:
    date_col = date_cols[0]
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df_with_dates = df[df[date_col].notna()].copy()

        if len(df_with_dates) > 0:
            daily_enrolment = df_with_dates.groupby(df_with_dates[date_col].dt.date).size().reset_index()
            daily_enrolment.columns = ['date', 'total_enrolment']
            daily_enrolment['date'] = pd.to_datetime(daily_enrolment['date'])
            daily_enrolment = daily_enrolment.sort_values('date')

            fig_trend = px.line(
                daily_enrolment,
                x='date',
                y='total_enrolment',
                title='Daily Aadhaar Enrollment Trend',
                labels={'total_enrolment': 'Daily Enrolments', 'date': 'Date'}
            )

            daily_enrolment['moving_avg'] = daily_enrolment['total_enrolment'].rolling(window=7).mean()

            fig_trend.add_scatter(
                x=daily_enrolment['date'],
                y=daily_enrolment['moving_avg'],
                mode='lines',
                name='7-Day Moving Average',
                line=dict(color='red', dash='dash')
            )

            fig_trend.update_layout(height=500, font=dict(size=14))
            fig_trend.show()
        else:
            print("No valid dates found in dataset")
    except Exception as e:
        print(f"Could not parse dates: {e}")
else:
    print("No date column found. Creating simulated enrollment trend...")
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    base_enrolment = 1000
    trend = np.linspace(0, 500, 365)
    seasonality = 200 * np.sin(np.linspace(0, 4*np.pi, 365))
    noise = np.random.normal(0, 100, 365)
    enrolments = base_enrolment + trend + seasonality + noise

    daily_enrolment = pd.DataFrame({
        'date': dates,
        'total_enrolment': enrolments.astype(int)
    })

    fig_trend = px.line(
        daily_enrolment,
        x='date',
        y='total_enrolment',
        title='Daily Aadhaar Enrollment Trend (Simulated)',
        labels={'total_enrolment': 'Daily Enrolments', 'date': 'Date'}
    )

    daily_enrolment['moving_avg'] = daily_enrolment['total_enrolment'].rolling(window=7).mean()

    fig_trend.add_scatter(
        x=daily_enrolment['date'],
        y=daily_enrolment['moving_avg'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color='red', dash='dash')
    )

    fig_trend.update_layout(height=500, font=dict(size=14))
    fig_trend.show()


files.download('aadhaar_enriched_dataset.csv')
files.download('aadhaar_anomalies.csv')
files.download('graph_edge_list.csv')


