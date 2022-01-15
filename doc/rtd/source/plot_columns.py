from teca import *
import numpy as _np
import matplotlib.pyplot as plt



class plot_columns(teca_python_algorithm):
    """ plots the columns of a teca_table """

    def __init__(self):
        self.x_axis = 'time'
        self.interactive = 0

    def set_interactive(self, val):
        """ Set interactive mode. Plots are displayed as they are generated. """
        self.interactive = val


    def set_x_axis_variable(self, x):
        """ Set the x-axis variable. """
        self.x_axis = x

    def execute(self, port, data_in, req):
        """ TECA pipeline execute phase. """

        table = as_teca_table(data_in[0])
        if table is None:
            return data_in[0]

        # get the x axis. it is returned as a teca-variant_array so we also get
        # a handle exposing the data as a Numpy array.
        x_ax = self.x_axis
        x_var = table.get_column(x_ax)
        hx_var = x_var.get_cpu_accessible()

        # loop over the columns
        i = 0
        n_col = table.get_number_of_columns()
        while i < n_col:
            # get the name of the ith column
            y_ax = table.get_column_name(i)

            i += 1

            # skip the x axis
            if y_ax == self.x_axis:
                continue

            # get the y axis. it is returned as a teca_variant_array so we also
            # get a handle exposing the data as a Numpy array.
            y_var = table.get_column(y_ax)
            hy_var = y_var.get_cpu_accessible()

            # plot the data
            fig = plt.figure()
            plt.plot(np.array(hx_var), np.array(hy_var))
            plt.xlabel(x_ax)
            plt.ylabel(y_ax)
            plt.title(title := '%s vs %s' % (y_ax, x_ax))
            plt.savefig(title.replace(' ', '_') + '.png')

        # show the plots.
        if self.interactive:
            plt.show()

        # pass the data downstream. this is optional but enables one to add
        # additional pipeline stages
        return data_in[0]



