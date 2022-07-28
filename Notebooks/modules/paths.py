import os


def paths_genesis(paths_dict, main_path='', main_name=''):

    # TODO: I am sure there is a more optimum way of doing this function (PRF).

    """ A function to create a whole file structure at a given path.

        Given a file structure represented by a dictionary and a specific path
        this function will create the complete structure in the given path.

        Parameters:
            main_path: (str)
                Path where the file structure is going to be created.
            paths_dict: (dict)
                The desired file structure.
            main_name: (str)
                The name to substitute the main_path in the return dictionary.

        Returns:
            paths: (dict)
                 Contains all the paths to all the created folders in such a way
                 that they can be called in an easy way.

        Example:
            folder_names = {'f1':['f1_sub1', 'f1_sub2'],
                           'f2':['f2_sub1', 'f2_sub2',{'f2_sub3': ['f2_sub3_sub1', 'f2_sub3_sub2']}],
                           'f3':['f3_sub1'],
                           'f4': None
                            }

            output_paths = {'f1' : path_to_f1,
                            'f1_sub1': path_to_f1_sub1,
                            'f2_sub1': path_to_f2_sub1,
                            'f2_sub3_sub1': path_to_f2_sub3_sub1,
                            'f4': path_to_f4,
                            'etc'
                            }
    """

    # Assigning the main_path as the working directory if one is not given
    if not main_path:
        main_path = os.getcwd()

    # Assigning a main_name if one is given
    if main_name:
        main_name = main_name + '_'

    paths = {}

    # Starting the creation of the folders
    for key in paths_dict:
        sub_path = os.path.join(main_path, key)
        os.makedirs(sub_path, exist_ok=True)
        reference_name = main_name + key

        # Verifying if the key is a meant to be a folder or a folder containing sub-folders and taking pertinent action

        # In case the key is meant to be just a folder
        if paths_dict[key] is None:
            paths[reference_name] = sub_path

        # In case the key is meant to be a folder containing sub-folders
        elif isinstance(paths_dict[key], list):
            paths[reference_name] = sub_path
            for folder_name in paths_dict[key]:
                # In case the sub-folder is a folder without sub-folders
                if isinstance(folder_name, str):
                    sub_sub_path = os.path.join(sub_path, folder_name)
                    os.makedirs(sub_sub_path, exist_ok=True)
                    sub_sub_name = reference_name + '_' + folder_name
                    paths[sub_sub_name] = sub_sub_path

                # In case the sub-folder contains more sub-folders
                elif isinstance(folder_name, dict):
                    sub_sub_name = reference_name
                    new_paths = paths_genesis(folder_name, sub_path, sub_sub_name)
                    paths.update(new_paths)

        # In case the key is supposed to be a folder containing a sub-folder with a more complex structure
        elif isinstance(paths_dict[key], dict):
            paths[reference_name] = sub_path
            sub_name = reference_name
            new_paths = paths_genesis(paths_dict[key], sub_path, sub_name)
            paths.update(new_paths)

        else:
            raise TypeError('One of the values in your dictionary is not type None, list or dict.')

    return paths


def read_structure(main_path, read='dir'):

    # TODO: This function has redundancies that might be better code. But they are not that big of a deal (PRF).

    """ Reads the folder/file structure of a given directory.

            Function that reads the structure of a folder and places it in a dictionary
            with the keys of the dictionary being the a name for the folder/file and the values
            the paths towards those folders/files.

            Parameters:
                main_path: (str)
                    The path for the folder in question.

                read: (str)
                    This parameters determines if the function is going to return the file
                    structure of the folder or the directory structure of the folder. The
                    two possible values are 'dir', 'file' or 'both'. Notice that 'both' is
                    not recommended.

            Returns:
                structure: (dict)
                    A dictionary with the keys being a name for the folders/files and the
                    value being the path towards those folders/files.
            """

    # Normalizing and getting the absolute path of the given main_path
    norm_path = os.path.abspath(os.path.normpath(main_path))
    # Extracting the directory name of the folder to form the keys of the dictionary
    dirname = os.path.dirname(norm_path)

    paths = {}

    # In case the user wants only the paths for the folders.
    if read == 'dir':
        for directory, folder_names, _ in os.walk(main_path):
            directory_key = directory.replace(dirname + os.sep, '').replace(os.sep, '_')
            paths[directory_key] = directory

            for folder in folder_names:
                folder_path = os.path.join(directory, folder)
                folder_key = folder_path.replace(dirname + os.sep, '').replace(os.sep, '_')
                paths[folder_key] = folder_path

    # In case the user wants only the paths for the files.
    elif read == 'file':
        for directory, _, file_names in os.walk(main_path):
            for file in file_names:
                file_path = os.path.join(directory, file)
                file_key = file_path.replace(dirname + os.sep, '').replace(os.sep, '_')
                paths[file_key] = file_path

    # In case the user wants both the paths for the folders and files.
    elif read == 'both':
        for directory, folder_names, file_names in os.walk(main_path):
            directory_key = directory.replace(dirname + os.sep, '').replace(os.sep, '_')
            paths[directory_key] = directory

            # Redundant, appears in the if 'dir' case
            for folder in folder_names:
                folder_path = os.path.join(directory, folder)
                folder_key = folder_path.replace(dirname + os.sep, '').replace(os.sep, '_')
                paths[folder_key] = folder_path

            # Redundant, appears in the if 'file' case
            for file in file_names:
                file_path = os.path.join(directory, file)
                file_key = file_path.replace(dirname + os.sep, '').replace(os.sep, '_')
                paths[file_key] = file_path

    # Catching errors
    else:
        raise Exception('Wrong value for parameter "read". Only "dir", "file" or "both" values are possible.')

    return paths
