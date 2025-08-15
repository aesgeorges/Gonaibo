from typing import List, Dict, Any, Optional
from shapely.geometry import Polygon
import netCDF4 as netcdf
import geopandas as gpd
import numpy as np


class Attenuation:
    """
       A class to handle importing ADCIRC exports across scenarios (and time horizons) and 'baseline' case for storm surge attenuation analysis over SLR projections. 
       Attenuation is computed as the difference in flooding between a given mangrove cover case and the baseline "no-mangrove" case.
    
    Attributes:
        mangrove_root (str): Root directory for mangrove scenarios
        nomangrove_root (str): Root directory for no-mangrove scenarios
        scenarios (List[str]): List of scenario names
        horizons (List[int]): List of time horizons in years    
    
    """

    def __init__(self,
                 mangrove_root: str,
                 nomangrove_root: str,
                 baseline_path: str, 
                 scenarios: List[str],
                 horizons: List[int] = [0],
                 mangrove_data: Optional[Dict] = None,
                 nomangrove_data: Optional[Dict] = None,
                 baseline_data: Optional[Any] = None,
             ):
        """
            Initialize the Atttenuation object for given baseline and other modeling scenarios.

            mangrove_root: root directory of scenarios with mangrove.
            nomangrove_path: root directory of scenarios without mangrove..
            baseline_path: path of baseline mangrove scenario.
            scenarios: names of different SLR projections.
            horizons: list of ints for SLR horizons.
            mangrove_data: dictionary to hold data for mangrove cases.
            nomangrove_data: dictionary to hold data no-mangrove cases.
            baseline_data: baseline mangrove case data.. 
        """

        self.baseline_path = baseline_path
        self.mangrove_root = mangrove_root
        self.nomangrove_root = nomangrove_root
        self.scenarios = scenarios
        self.horizons = horizons
        self.mangrove_data = {}
        self.nomangrove_data = {}
        self.baseline_data = None
        
        # TODO : add check for valid paths.
        
        # Load data into Dictionary
        self.load_all()


    def load_all(self):
        # Load mangrove data
        self.mangrove_data = self.load_data(self.mangrove_root)
        self.nomangrove_data = self.load_data(self.nomangrove_root)
        
        
    def load_data(self, root: str) -> Dict:
        # Load empty dictionary to be populated across scenarios/horizons
        data = {}
        failed_loads = []

        for scenario in self.scenarios:
            data[scenario] = {}
            for horizon in self.horizons:
                path = f'{root}{scenario}_{horizon}/hotstart/maxele.63.nc'
                try:
                    dataset = netcdf.Dataset(path, 'r')
                    data[scenario][horizon] = {
                        'zeta_max': dataset['zeta_max'][:],
                        'x': dataset['x'][:],
                        'y': dataset['y'][:],
                        'element': dataset['element'][:],
                        'depth': dataset['depth'][:],
                        'fill_value': dataset['zeta_max']._FillValue,
                        'attrs': {attr: getattr(dataset, attr) for attr in dataset.ncattrs()}
                    }
                    dataset.close()
                except FileNotFoundError:
                    print(f"Error: Could not find file at {path}")
                    data[scenario][horizon] = None
                    failed_loads.append(path)

        if failed_loads:
            print(f"Warning: Failed to load {len(failed_loads)} files")
        return data


    def compute_scenarios_protection(self):
        """
           Compute the protection effect of storm surge attenuation for all cover and no cover scenarios loaded in Attenuation Class. 
        """

        data_protection = {}      
        # Iterate through scenarios and horizons  
        for scenario in self.scenarios:
            data_protection[scenario] = {}
            for year in self.horizons:
                # Load no mangrove and mangrove data
                data_m = self.mangrove_data[scenario][year]
                data_nm = self.nomangrove_data[scenario][year]
                # Compute protection for given scenario
                data_protection[scenario][year] = self.compute_max_protection(data_m, data_nm)
        return data_protection


    def compute_max_protection(self, data_mangrove: netcdf.Dataset, data_nomangrove: netcdf.Dataset) -> Dict:
        """
        Compute the protection effect of mangroves by taking the spatial difference in zeta_max (maximum WSE) between mangrove and no-mangrove scenarios.
        Returns a dictionary with the protection results.
        """
        
        # Load zeta_max variable for mangrove and no mangrove cases
        zeta_mangrove = data_mangrove['zeta_max'][:]
        zeta_nomangrove = data_nomangrove['zeta_max'][:]

        # Load fill value for NaNs and convert to zero for later operation
        fill_value = data_mangrove['fill_value']
        zeta_m_calc = np.ma.filled(zeta_mangrove, 0)
        zeta_nm_calc = np.ma.filled(zeta_nomangrove, 0)

        # Calculate Flooding Difference/Protection
        # Protection is nomangrove - mangrove: subtract from extra flooding so that protection comes out positive.
        protection = zeta_nm_calc - zeta_m_calc

        # Make mask where both original datasets had fill values
        both_masked = np.ma.getmaskarray(zeta_mangrove) & np.ma.getmaskarray(zeta_nomangrove)

        # Restore fill values using mask
        protection[both_masked] = fill_value
        protection = np.ma.masked_where(both_masked, protection, copy=False)
        protection.set_fill_value(fill_value)

        results = {}
        results['x'] = data_mangrove['x'][:]
        results['y'] = data_mangrove['y'][:]
        results['element'] = data_mangrove['element'][:]
        results['depth'] = data_mangrove['depth'][:]
        results['zeta_mangrove'] = zeta_mangrove
        results['zeta_nomangrove'] = zeta_nomangrove
        results['protection'] = protection
        results['attrs'] = data_mangrove['attrs']
        
        return results


    def calculate_flooded_area(self, center_lon: float, center_lat: float, data: Dict) -> float:
        """
            Calculate flooded area using geopandas with equal-area projection.

            Attributes:
            center_lon (float): Longitude of centered location for reprojection.
            center_lat (float): Latitude of centered location for reprojection.
        """
        flooded_polygons = []
        
        # Pulling mesh data from Dict
        x = data['x'][:]
        y = data['y'][:]
        elements = data['element'][:]
        
        # Make overland flooding masks
        ### TODO: check this logic, this will not cover inland areas under mSL that are 100% getting flooded if the surge reaches.
        land_mask = data['depth'][:] < 0
        flood_mask = (data['zeta_max'][:] > 0) & land_mask

        # Build polygon of flooded areas.
        for elem in elements:
            v1, v2, v3 = elem - 1 # Adjust for ADCIRC's one-based indexing
            
            if flood_mask[v1] or flood_mask[v2] or flood_mask[v3]:
                triangle = Polygon([(x[v1], y[v1]), 
                                    (x[v2], y[v2]),
                                    (x[v3], y[v3])])
                if triangle.is_valid:
                    flooded_polygons.append(triangle)

        if len(flooded_polygons) == 0:
            return 0

        # Create geodataframe for later calculations.
        gdf = gpd.GeoDataFrame(geometry=flooded_polygons, crs='EPSG:4326')

        # Use Lambert Azimuthal Equal Area projection at center-lat-lon for accurate area calculation
        equal_area_proj = f'+proj=laea +lat_0={center_lat} +lon_0={center_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

        # Convert geodf to equal area proj
        gdf_projected = gdf.to_crs(equal_area_proj)

        # Calculate total area of flooded polygons in km2
        total_area_km2 = gdf_projected.geometry.area.sum() / 1e6

        return total_area_km2        
