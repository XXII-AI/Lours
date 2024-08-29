import socketserver
from argparse import ArgumentParser
from logging import warn
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING

from watchdog.events import (
    EVENT_TYPE_CLOSED,
    EVENT_TYPE_OPENED,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer
from watchdog.observers.api import EventQueue

from lours.dataset import Dataset, from_caipy, from_caipy_generic
from lours.utils import try_import_fiftyone
from lours.utils.difftools import dataset_diff

if TYPE_CHECKING:
    import fiftyone as fo
else:
    fo = try_import_fiftyone()


class DatasetUpdateHandler(FileSystemEventHandler):
    """Class to update the dataset in fiftyone each time a change is detected"""

    def __init__(
        self,
        dataset: Dataset,
        fo_dataset: fo.Dataset,
        images_root: Path,
        annotations_root: Path,
        splits_to_read: list[str],
        event_queue: EventQueue,
        cooldown_time: float = 1,
    ) -> None:
        """Constructor of DatasetUpdateHandler

        The handler will wait for a cooldown period of time and then will cancel all
        changes detected by watchdog in the mean time. This allows to only run the
        dataset update once when multiple files are modified at the same time.

        Args:
            dataset: dataset object to compare the newly loaded caipy dataset with,
                should have already been converted to fiftyone
            fo_dataset: fiftyone dataset object corresponding to the lours dataset.
                will be used to remove modified or deleted data from the caipy folder.
            images_root: parameter given to caipy loading function, should be the same
                as the one used when constructing the already existing dataset
            annotations_root: parameter given to caipy loading function, should be the
                same as the one used when constructing the already existing dataset
            splits_to_read: parameter given to caipy loading function, should be the
                same as the one used when constructing the already existing dataset
            event_queue: queue object to know if other changes have been detected after
                the cooldown
            cooldown_time: time in seconds between a detected change and a dataset
                update. Every change event between first event and the cooldown will be
                ignored. Useful when adding multiple files at once. Defaults to 1
        """
        super().__init__()
        self.dataset = dataset
        self.fo_dataset = fo_dataset
        self.images_root = images_root
        self.annotations_root = annotations_root
        self.splits_to_read = splits_to_read
        self.cooldown_time = cooldown_time
        self.event_queue = event_queue

    def update_dataset(self) -> None:
        """Compare the dataset with the one loaded from caipy.

        If different, add the new elements to fiftyone, but only them.
        """
        caipy_dataset = from_caipy_generic(
            images_folder=self.images_root,
            annotations_folder=self.annotations_root,
            splits_to_read=self.splits_to_read,
        )
        to_update, to_remove, common = dataset_diff(caipy_dataset, self.dataset)
        if len(to_remove) > 0:
            annotations_filter = fo.ViewField("lours_id").is_in(
                common.annotations.index
            )
            images_filter = fo.ViewField("lours_id").is_in(common.images.index)

            filtered_view = self.fo_dataset.match(images_filter)
            if "groundtruth_detection" in self.fo_dataset.get_field_schema():
                filtered_view = filtered_view.filter_labels(
                    "groundtruth_detection.detections", annotations_filter
                )
            if "groundtruth_keypoints" in self.fo_dataset.get_field_schema():
                filtered_view = filtered_view.filter_labels(
                    "groundtruth_keypoint.keypoints", annotations_filter
                )
            # Register this view on top of the existing dataset
            filtered_view.save()
            filtered_view.keep()
            filtered_view.keep_fields()

        if len(to_update) > 0:
            self.fo_dataset = to_update.to_fiftyone(
                dataset_name=str(self.fo_dataset.name), existing="update"
            )
        self.fo_dataset.save()
        self.dataset = caipy_dataset

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Method called when a file or folder is created, modified or deleted.

        If after a cooldown poeriod, the event queue has elements, this will remove
        them from the queue so that the dataset update is not called again. Since the
        dataset update reads the whole folder, the subsequent detected changes will not
        be ignored even though they are removed from the event queue.

        Args:
            event: event describing the file or folder that was created.
                Not used today, but could probably be used in the future to avoid
                unnecessary reloading of caipy dataset
        """
        if event.event_type in [EVENT_TYPE_CLOSED, EVENT_TYPE_OPENED]:
            return
        sleep(self.cooldown_time)
        while not self.event_queue.empty():
            self.event_queue.get()
        self.update_dataset()


def get_argparser() -> ArgumentParser:
    """Function to get the argparser, which will be parsed itself by sphinx-argparse"""
    parser = ArgumentParser(
        description=(
            "Directly convert a Caipy folder into fiftyone for quick inspection"
        ),
    )

    i_parser = parser.add_argument_group("Input arguments")
    i_parser.add_argument(
        "--mode",
        choices=["vanilla", "generic"],
        default="vanilla",
        help=(
            "Choice between vanilla mode, where a single folder is given with option"
            " ``--input_folder``, and generic, where two folders are given with options"
            " ``--images_root`` and ``--annotations_root``"
        ),
    )
    i_parser.add_argument(
        "--input-folder",
        "-i",
        help="When in vanilla mode, folder where the CAIPY dataset is stored",
        type=Path,
        default=None,
    )
    i_parser.add_argument(
        "--images_root",
        "--ir",
        help=(
            "When in generic mode, folder where images are stored. Equivalent folder in"
            " vanilla mode is ``dataset/Images``"
        ),
        type=Path,
        default=None,
    )
    i_parser.add_argument(
        "--annotations_root",
        "--ar",
        help=(
            "When in generic mode, folder where annotations json files are stored."
            " Equivalent folder in vanilla mode is ``dataset/Annotations``"
        ),
        type=Path,
        default=None,
    )
    i_parser.add_argument(
        "--splits-to-read",
        "-s",
        nargs="*",
        default=None,
        help=(
            "Optional list of splits to read. If not selected, will read all splits and"
            " convert them to fiftyone"
        ),
    )
    f_parser = parser.add_argument_group("Fiftyone arguments")
    f_parser.add_argument(
        "--dataset-name",
        "--name",
        "-n",
        help=(
            "Optional dataset name to appear in fiftyone app. If not selected, will"
            " take the name of loaded dataset object, i.e. the name of ``input-folder``"
            " or parent of ``--images-root`` without its parents"
        ),
        default=None,
    )
    f_parser.add_argument(
        "--not-persistent",
        "--np",
        action="store_false",
        dest="persistent",
        help=(
            "If selected, will not save the converted dataset in fiftyone. It will be "
            "available to inspect with the browser during the time this script is"
            "running, but will be removed as soon as the script is stopped."
        ),
    )
    f_parser.add_argument(
        "--no-app",
        "--na",
        action="store_false",
        dest="launch_app",
        help=(
            "If selected, will not launch the app. This needs the 'persistent' option"
            " to be selected for the command to do anything"
        ),
    )
    f_parser.add_argument(
        "--watch",
        "-w",
        action="store_true",
        help=(
            "If selected, will watch the folder containing the dataset. Each time a"
            " change in the folder is detected, the dataset is updated and so is the"
            " fiftyone counterpart. Useful when you are constructing a dataset"
            " gradually. Note that this option is useless if the app is not launched"
        ),
    )
    f_parser.add_argument(
        "--port",
        "-p",
        default=0,
        type=int,
        help=(
            "Server port to connect to the app server. If not set, will choose randomly"
            " a free port"
        ),
    )
    f_parser.add_argument(
        "--only-local",
        action="store_true",
        help=(
            "if selected, will not open the server for other than localhost. the server"
            " can still be reachable with ssh tunnelling though"
        ),
    )
    return parser


def run():
    """CLI function for caipy to fiftyone"""
    parser = get_argparser()

    args = parser.parse_args()

    if not args.launch_app and not args.persistent:
        warn(
            "App won't be launched and dataset is not persistent, this command will not"
            " do anything",
            RuntimeWarning,
        )

    if args.mode == "vanilla":
        assert (
            args.input_folder is not None
        ), "You must provide a path to input_folder for vanilla mode"
        annotations_root = args.input_folder / "Annotations"
        images_root = args.input_folder / "Images"
        dataset = from_caipy(args.input_folder, splits_to_read=args.splits_to_read)
    else:
        assert args.images_root is not None and args.annotations_root is not None, (
            "You must provide paths for both images_root and annotations_root for"
            " generic mode"
        )
        images_root = args.images_root
        annotations_root = args.annotations_root
        dataset = from_caipy_generic(
            images_folder=args.images_root,
            annotations_folder=args.annotations_root,
            splits_to_read=args.splits_to_read,
        )

    dataset_name = (
        args.dataset_name if args.dataset_name is not None else dataset.dataset_name
    )

    fo_dataset = dataset.to_fiftyone(dataset_name=dataset_name)
    if args.persistent:
        fo_dataset.persistent = True

    if args.launch_app:
        launch_kwargs = {}
        if not args.only_local:
            launch_kwargs["address"] = "0.0.0.0"
        if args.port == 0:
            print("Getting a random free port ...")
            # Stolen from https://stackoverflow.com/a/61685162
            with socketserver.TCPServer(("localhost", 0), None) as s:  # pyright: ignore
                port = s.server_address[1]
            print(f"Port chosen : {port}")
        else:
            port = args.port
        launch_kwargs["port"] = port
        session = fo.launch_app(dataset=fo_dataset, **launch_kwargs)
        if args.watch:
            observer = Observer()
            event_handler = DatasetUpdateHandler(
                dataset,
                fo_dataset,
                images_root,
                annotations_root,
                args.splits_to_read,
                observer.event_queue,
            )
            observer.schedule(event_handler, str(annotations_root), recursive=True)
            observer.start()
        session.wait(-1)
