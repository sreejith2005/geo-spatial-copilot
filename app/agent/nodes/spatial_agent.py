import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import rasterio
from app.agent.state import AgentState



def analyze_spatial(state: AgentState) -> dict:
    """
    Spatial Agent node.
    Looks up chips in the spatial catalog based on the query or chip_path.
    """
    query = state.get("user_query", "").lower()
    chip_path = state.get("chip_path")
    
    catalog_path = "data/chip_catalog.parquet"
    spatial_result = []
    reasoning = []
    tools_used = []
    
    if os.path.exists(catalog_path):
        tools_used.append("Spatial Parquet Catalog")
        df = pd.read_parquet(catalog_path)
        
        if chip_path:
            reasoning.append(f"Looking up spatial metadata for specific chip: {chip_path}")
            # normalize path
            chip_path_norm = chip_path.replace('\\', '/')
            matches = df[df["file_path"].str.contains(chip_path_norm, case=False, na=False, regex=False)]
            if not matches.empty:
                chip_data = matches.iloc[0].to_dict()
                
                district = chip_data.get('district', 'Unknown')
                state_loc = chip_data.get('state', 'Unknown')
                reasoning.append(f"Loaded DataMeet boundaries from catalog: {district}, {state_loc}")
                tools_used.append("DataMeet Boundaries")
                    
                spatial_result.append(chip_data)
                reasoning.append(f"Found spatial metadata for {chip_path}")
            else:

                reasoning.append(f"No spatial metadata found in catalog for {chip_path}")
                
        else:
            reasoning.append("Searching for relevant spatial regions based on query.")
            # Simple keyword matching against country or region
            matches = []
            for _, row in df.iterrows():
                country = str(row.get("country", "")).lower()
                region = str(row.get("region", "")).lower()
                if country and country in query:
                    matches.append(row.to_dict())
                elif region and region in query:
                    matches.append(row.to_dict())
                    
            if matches:
                # Prioritize Sentinel-1 SAR chips over label masks for vision compatibility
                s1_matches = [m for m in matches if "_S1Hand.tif" in str(m.get("file_path", ""))]
                selected_matches = s1_matches if s1_matches else matches
                
                reasoning.append(f"Found {len(selected_matches)} chips matching regions in query.")
                for m in selected_matches[:3]:
                    spatial_result.append(m)
            else:
                reasoning.append("No specific region matched in catalog. Taking a sample.")
                spatial_result.append(df.iloc[0].to_dict())
                
    else:
        reasoning.append("Spatial catalog not found. Skipping spatial intelligence.")
        
    return {
        "spatial_result": spatial_result,
        "executed_agents": ["spatial_agent"],
        "reasoning_path": reasoning,
        "tools_used": tools_used,
        "trace_log": ["Executed Spatial Agent."]
    }
