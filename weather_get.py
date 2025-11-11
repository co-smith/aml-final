import cdsapi

# This connects to the Climate Data Store
client = cdsapi.Client()

print("Requesting ERA5 data for NYC...")

client.retrieve(
    # This is the standard dataset for ML weather training
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf', # NetCDF is much easier to work with in Python (using xarray)
        'variable': [
            '2m_temperature', 
            '10m_u_component_of_wind', 
            '10m_v_component_of_wind',
            'surface_pressure',
            '2m_dewpoint_temperature',
        ],
        # A bounding box for the NYC area [North, West, South, East]
        # This will give us data on a 0.25-degree grid
        'area': [
            41.0, -74.5,
            40.0, -73.5,
        ],
        'year': '2023',
        'month': [
            # '04', '05', '06',
             '10', '11', '12',
        ],
        'day': [
            '01', '02', '03', '04', '05', '06',
            '07', '08', '09', '10', '11', '12',
            '13', '14', '15', '16', '17', '18',
            '19', '20', '21', '22', '23', '24',
            '25', '26', '27', '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
        ],
    },
    'nyc_era5_data_q4.nc') # The file will be saved as a NetCDF

print("Download complete: nyc_era5_data.nc")
