
def CreateDataLoader(opt,
                    horizontal_utilization_map,
                    vertical_utilization_map,
                    pin_density_map):
    from .custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt, horizontal_utilization_map, vertical_utilization_map, pin_density_map)
    return data_loader
