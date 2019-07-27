from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self, log_dir, comment=None):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        self.comment = comment
        if comment is None:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = SummaryWriter(log_dir, comment=comment)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for pair in tag_value_pairs:
            tag, value = pair
            self.scalar_summary(tag, value, step)

    def image_summary(self, tag, images, step):
        """
        batch of images should be a 4d-tensor torchvision vector use make_grid
        single image 3d-tensor (3, H, W)
        ** images should be normalized
        """

        # add image
        self.writer.add_image(tag, images, step)

    def log_histogram(self, tag, values, step, bins=100):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)
        fig = plt.hist(values, bins=bins)
        plt.title("historgram")

        # add histogram
        self.writer.add_figure(tag, fig, step)
