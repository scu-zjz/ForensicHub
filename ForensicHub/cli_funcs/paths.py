
import os
from pathlib import Path
import appdirs

class ForHubPath:
    app_name = "ForensicHub"
    app_author = "ForensicHub-authors"

    @classmethod
    def get_data_storage_path(cls):
        storage_path = appdirs.user_data_dir(cls.app_name, cls.app_author)
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        return storage_path

    @classmethod
    def get_package_dir(cls):
        return Path(__file__).parent.parent

    @classmethod
    def get_templates_dir(cls):
        return cls.get_package_dir() / 'training_scripts'
    
    @classmethod
    def get_train_test_yaml_dir(cls):
        return cls.get_package_dir() / 'statics'

    

# print("get templates dir: ", ForHubPath.get_templates_dir())