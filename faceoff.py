#!/usr/bin/python3

import argparse
import face_recognition as fr
import os
import shutil
import pickle
import signal
import sys
import time
import concurrent.futures


class FaceOff:

    # Only jpg, else I get too many garbage images
    image_file_types = ('.jpg', '.jpeg')
    image_files = []
    substractPath = '/mnt/8ac50e64-6998-4105-ab73-614e1ac8481c/Syncthing/Pixel XX/'

    def __init__(self, _options):
        self.source_directory = os.path.abspath(_options.source)
        if not os.path.isdir(self.source_directory):
            print('ERROR: source directory must exist')
            sys.exit(1)

        # Ugly and not a really good fix to still scan if the root folder doesn't have images, because subfolders can have them
        if not any(filename.endswith(self.image_file_types) for filename in os.listdir(self.source_directory)) and not _options.recursive:
            print('ERROR: source directory must contain image files')
            sys.exit(1)

        self.ignore = _options.ignore
        self.alone = _options.alone

        if _options.target:
            self.target_directory = os.path.abspath(_options.target)
        else:
            parser.print_help()
            print('ERROR: target directory must be specified')
            sys.exit(1)
        if not os.path.isdir(self.target_directory):
            print('ERROR: target directory must exist')
            sys.exit(1)

        if os.path.exists('./face_encodings.pkl') and not self.alone:
            with open('face_encodings.pkl', 'rb') as input_file:
                self.processed_face_encodings = pickle.load(input_file)
        else:
            self.processed_face_encodings = []

        if os.path.exists('./face_directories.pkl') and not self.alone:
            with open('face_directories.pkl', 'rb') as input_file:
                self.processed_face_directories = pickle.load(input_file)
        else:
            self.processed_face_directories = []

        # Keep track of processed files so a rerun doesn't reanalyse everything
        if os.path.exists('./face_processed_files.pkl') and not self.alone:
            with open('face_processed_files.pkl', 'rb') as input_file:
                self.processed_files = pickle.load(input_file)
        else:
            self.processed_files = []

    def process_image(self, file):
        print(f'Started processing {file}...')
        try:
            image = fr.load_image_file(file)
            face_locations = fr.face_locations(image, number_of_times_to_upsample=0, model='cnn')
            face_encodings = fr.face_encodings(image, face_locations)
            print(f'Processed {file}...')
            return file, face_encodings

        except Exception as err:
            print('ERROR: %s' % err)
            with open('/home/wout/.latest_faceoff_error.pkl', 'wb') as output_file:
                pickle.dump(err, output_file, pickle.HIGHEST_PROTOCOL)

    def run(self, _options):
        if _options.recursive:
            for root, directories, files in os.walk(self.source_directory):
                for file in files:
                    path = os.path.join(root, file)
                    if file.endswith(self.image_file_types) and path[len(self.substractPath):] not in self.processed_files:
                        self.image_files.append(path)
        else:
            for file in os.listdir(self.source_directory):
                path = os.path.join(self.source_directory, file)
                if file.endswith(self.image_file_types) and path[len(self.substractPath):] not in self.processed_files:
                    self.image_files.append(path)

        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            print('Running jobs...')
            results = executor.map(self.process_image, self.image_files)

            for file, face_encodings in results:
                if len(face_encodings) == 0 and not self.ignore:
                    if not os.path.exists(os.path.join(self.target_directory, 'no_face_found')):
                        os.mkdir(os.path.join(self.target_directory, 'no_face_found'))
                    shutil.copyfile(file, os.path.join(self.target_directory, 'no_face_found', os.path.basename(file)))
                else:
                    for face_encoding in face_encodings:
                        matches = fr.compare_faces(self.processed_face_encodings, face_encoding)
                        if True in matches:
                            # Fixes in case face encoding matches multiple faces
                            for match_index in range(len(matches)):
                                if matches[match_index] == True:
                                    face_id = self.processed_face_directories[match_index]
                                    if not os.path.exists(os.path.join(self.target_directory, face_id)):
                                        os.mkdir(os.path.join(self.target_directory, face_id))
                                    # Fix copy file asap here or else it probably gets lost
                                    shutil.copyfile(file, os.path.join(self.target_directory, face_id, os.path.basename(file)))
                        else:
                            face_id = 'face%d' % (len(self.processed_face_encodings) + 1)
                            if not os.path.exists(os.path.join(self.target_directory, face_id)):
                                os.mkdir(os.path.join(self.target_directory, face_id))
                            self.processed_face_encodings.append(face_encoding)
                            self.processed_face_directories.append(face_id)
                            shutil.copyfile(file, os.path.join(self.target_directory, face_id, os.path.basename(file)))

                path = os.path.dirname(file) + '/' + os.path.basename(file)
                self.processed_files.append(path[len(self.substractPath):])
                # Write asap pkl data files in case analysis gets aborted it can rerun
                if not self.alone:
                    with open('face_encodings.pkl', 'wb') as output_file:
                        pickle.dump(self.processed_face_encodings, output_file, pickle.HIGHEST_PROTOCOL)
                    with open('face_directories.pkl', 'wb') as output_file:
                        pickle.dump(self.processed_face_directories, output_file, pickle.HIGHEST_PROTOCOL)
                    with open('face_processed_files.pkl', 'wb') as output_file:
                        pickle.dump(self.processed_files, output_file, pickle.HIGHEST_PROTOCOL)


def exit_gracefully(_signal, _frame):
    print('\rGoodbye!')
    exit()


if __name__ == '__main__':
    start = time.perf_counter()
    VERSION = '20191108_2331'
    desc = """
****************************************
 FaceOff - Quick and dirty image sorter
 Version %s
 Author: greatwhitehat (c) 2018
****************************************
""" % VERSION

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=desc)
    parser.add_argument('--source', help='Source directory')
    parser.add_argument('--target', help='Target directory')
    parser.add_argument('--recursive', default=False, action='store_true', help='Enable recursive processing of '
                                                                                'sub-directories of the source')
    parser.add_argument('--ignore', default=False, action='store_true', help='Do not copy images where no faces have '
                                                                             'been detected')
    parser.add_argument('--alone', default=False, action='store_true', help='Do not remember this work for future '
                                                                            'runs and do not load prior runs.')
    options = parser.parse_args()

    if not (options.source and options.target):
        parser.print_help()
        print('ERROR: source and target directory must be specified')
        sys.exit(1)

    # Catch CTRL+C and exit in a nice way
    signal.signal(signal.SIGINT, exit_gracefully)

    face_off = FaceOff(options)
    face_off.run(options)

    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)')
