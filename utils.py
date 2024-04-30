import gzip
import array
import struct
import jax
import equinox as eqx
from einops import rearrange
from os import path


def get_mnist_data(path_):
    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return jax.numpy.array(array.array("B", fh.read()), dtype=jax.numpy.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return jax.numpy.array(
                array.array("B", fh.read()), dtype=jax.numpy.uint8
            ).reshape(num_data, rows, cols)

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:

        train_images = parse_images(path.join(path_, "train-images-idx3-ubyte.gz"))
        train_labels = parse_labels(path.join(path_, "train-labels-idx1-ubyte.gz"))
        test_images = parse_images(path.join(path_, "t10k-images-idx3-ubyte.gz"))
        test_labels = parse_labels(path.join(path_, "t10k-labels-idx1-ubyte.gz"))

    return (
        train_images,
        train_labels,
        test_images,
        test_labels,
    )


def preprocess(x):
    x = x / 255.0
    x = rearrange(x, "... h w -> ... (h w)")
    x = x / jax.numpy.sqrt((x**2).sum(-1, keepdims=True))
    return x


class VectorizedHAM(eqx.Module):
    _ham: eqx.Module

    def __init__(self, ham):
        self._ham = ham

    @property
    def _batch_axes(self):
        """
        A helper function to tell vmap to batch along the 0'th dimension of each state in the HAM.
        """
        return {k: 0 for k in self._ham.neurons.keys()}

    def init_neuron_states(self, bs=None):
        return self._ham.init_neuron_states(bs)

    def activate(self, xs):
        return jax.vmap(self._ham.activate, in_axes=(self._batch_axes,))(xs)

    def dEdg(self, gs, xs, return_energy=False):
        return jax.vmap(
            self._ham.dEdg, in_axes=(self._batch_axes, self._batch_axes, None)
        )(gs, xs, return_energy)

    def energy(self, gs, xs):
        return jax.vmap(self._ham.energy, in_axes=(self._batch_axes, self._batch_axes))(
            gs, xs
        )

    def energy_tree(self, gs, xs):
        return jax.vmap(
            self._ham.energy_tree, in_axes=(self._batch_axes, self._batch_axes)
        )(gs, xs)

    def connection_energies(self, gs):
        return jax.vmap(self._ham.connection_energies, in_axes=(self._batch_axes,))(gs)

    def neuron_energies(self, gs, xs):
        return jax.vmap(
            self._ham.neuron_energies, in_axes=(self._batch_axes, self._batch_axes)
        )(gs, xs)

    def unvectorize(self):
        return self._ham

    def vectorize(self):
        return self

    @property
    def synapses(self):
        return self._ham.synapses


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from einops import rearrange


# set the colormap and centre the colorbar
class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


cnorm = MidpointNormalize(midpoint=0.0)


def show_img(img):
    vmin, vmax = img.min(), img.max()
    vscale = max(np.abs(vmin), np.abs(vmax))
    cnorm = MidpointNormalize(midpoint=0.0, vmin=-vscale, vmax=vscale)

    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    pcm = ax.imshow(img, cmap="seismic", norm=cnorm)
    ax.axis("off")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
    fig.colorbar(pcm, cax=cbar_ax)
    return fig
