```mermaid
%%{init: {'theme':'default'}}%%
graph TD
    A[Data Collection & Upload] --> B[Data Preprocessing Pipeline]
    
    B --> B1[Raw CSV Loading]
    B1 --> B2[Data Cleaning & Deduplication]
    B2 --> B3[Column Standardization]
    B3 --> B4[Missing Value Imputation]
    
    B4 --> C[Feature Engineering Engine]
    
    C --> C1[Coverage Features]
    C1 --> C1a[Enrollment-Population Ratio]
    C1 --> C1b[Regional Coverage Metrics]
    
    C --> C2[Location Features]
    C2 --> C2a[Update Frequency Analysis]
    C2 --> C2b[Geographic Dispersion]
    
    C --> C3[Duplicate Detection Features]
    C3 --> C3a[Biometric Hashing & Matching]
    C3 --> C3b[Demographic Similarity Scoring]
    
    C1 --> D[Coverage Analysis Module]
    C2 --> E[Anomaly Detection Module]
    C3 --> F[Fraud Detection Module]
    
    D --> D1[Statistical Benchmarking]
    D1 --> D2[Coverage Visualization]
    D2 --> D2a[Regional Coverage Charts]
    D2 --> D2b[Population vs Enrollment Plots]
    D2 --> D2c[Coverage Heatmaps]
    
    E --> E1[Isolation Forest Algorithm]
    E1 --> E2[Anomaly Flagging]
    E2 --> E3[Anomaly Visualization]
    E3 --> E3a[Scatter Plots]
    E3 --> E3b[Box Plot Analysis]
    
    F --> F1[Multi-modal Detection]
    F1 --> F1a[Biometric Matching]
    F1a --> F1b[Demographic Pattern Analysis]
    F1b --> F1c[Graph-based Similarity]
    
    F1c --> F2[Clustering Engine]
    F2 --> F2a[K-Means Clustering]
    F2a --> F2b[PCA Dimensionality Reduction]
    F2b --> F3[Risk Scoring]
    F3 --> F4[Cluster Visualization]
    F4 --> F4a[Network Graphs]
    F4 --> F4b[PCA Scatter Plots]
    F4 --> F4c[Cluster Distribution Charts]
    
    D2 --> G[Interactive Dashboard System]
    E3 --> G
    F4 --> G
    
    G --> G1[KPI Metrics Dashboard]
    G1 --> G2[Interactive Visualizations]
    G2 --> G2a[Coverage Maps]
    G2 --> G2b[Anomaly Explorer]
    G2 --> G2c[Cluster Visualizer]
    G2 --> G2d[Risk Heatmaps]
    
    G2 --> H[Network Graph Construction]
    H --> H1[Heterogeneous Graph]
    H1 --> H1a[Person Nodes]
    H1 --> H1b[Location Nodes]
    H1 --> H1c[Biometric Nodes]
    H1 --> H1d[Similarity Edges]
    
    H1 --> I[Results Export System]
    I --> I1[Enriched Dataset CSV]
    I1 --> I2[Anomalies Export]
    I2 --> I3[Graph Edge List]
    I3 --> I4[File Download]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#ffebee
    style F fill:#f3e5f5
    style G fill:#e8eaf6
    style H fill:#fce4ec
    style I fill:#e0f2f1
