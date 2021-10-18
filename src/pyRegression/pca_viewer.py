# -*- coding: utf-8 -*-

import numpy as np
from gui_objects.plot_canvas import plot_canvas
from PyQt5.QtWidgets import (QMainWindow, QLabel, QComboBox, QWidget,
                             QGridLayout, QHBoxLayout, QVBoxLayout,
                             QDesktopWidget)
# from pyAnalytics.spectroscopy_data import spectroscopy_data
# from pyAnalytics.raman_data import raman_image


class pca_viewer(QMainWindow):
    def __init__(self, pca_data):
        super().__init__()
        self.init_window()
        self.define_widgets()
        self.position_widgets()

        self.plots_active = True
        self.update_data(pca_data)
        self.connect_event_handlers()

    def init_window(self):
        self.setGeometry(500, 500, 1200, 900)  # xPos, yPos, width, heigth
        self.center()  # center function is defined below
        self.setWindowTitle('PCA viewer')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.select_column_combo = QComboBox()

        self.scores_plot = plot_canvas(
            plot_title='Scores plot', x_axis_title='PC score')
        self.loadings_plot = plot_canvas(
            plot_title='Loadings plot', x_axis_title='wavenumber [cm-1]')
        self.explained_variance_plot = plot_canvas(
            plot_title='Explained variance', x_axis_title='PC number')
        self.reconstructed_spectra_plot = plot_canvas(
            plot_title='Reconstructed data',
            x_axis_title='wavenumber [cm-1]')

        self.x_coord_label = QLabel('x')
        self.x_coord_combo = QComboBox()
        self.y_coord_label = QLabel('y')
        self.y_coord_combo = QComboBox()
        self.z_coord_label = QLabel('z')
        self.z_coord_combo = QComboBox()
        self.sample_label = QLabel('Sample')
        self.sample_combo = QComboBox()

        self.pcs_for_reconstruction_label = QLabel('PCs for reconstruction')
        self.pcs_for_reconstruction_combo = QComboBox()

        self.first_pc_scores_combo = QComboBox()
        self.second_pc_scores_combo = QComboBox()

        self.pc_loadings_combo = QComboBox()

    def position_widgets(self):
        self.pc_selection_scores_layout = QHBoxLayout()
        self.pc_selection_scores_layout.addWidget(self.first_pc_scores_combo)
        self.pc_selection_scores_layout.addWidget(self.second_pc_scores_combo)

        self.coord_combos_layout = QHBoxLayout()
        self.coord_combos_layout.addWidget(self.x_coord_label)
        self.coord_combos_layout.addWidget(self.x_coord_combo)
        self.coord_combos_layout.addWidget(self.y_coord_label)
        self.coord_combos_layout.addWidget(self.y_coord_combo)
        self.coord_combos_layout.addWidget(self.z_coord_label)
        self.coord_combos_layout.addWidget(self.z_coord_combo)
        self.coord_combos_layout.addWidget(self.sample_label)
        self.coord_combos_layout.addWidget(self.sample_combo)
        self.coord_combos_layout.addWidget(self.pcs_for_reconstruction_label)
        self.coord_combos_layout.addWidget(self.pcs_for_reconstruction_combo)
        self.coord_combos_layout.addStretch(1)

        self.reconstructed_spectra_layout = QVBoxLayout()
        self.reconstructed_spectra_layout.addWidget(
            self.reconstructed_spectra_plot)
        self.reconstructed_spectra_layout.addLayout(self.coord_combos_layout)

        self.grid_container.addLayout(self.pc_selection_scores_layout,
                                      *(0, 0), 1, 1)
        self.grid_container.addWidget(self.pc_loadings_combo, *(0, 1), 1, 1)

        self.grid_container.addWidget(self.scores_plot, *(1, 0), 1, 1)
        self.grid_container.addWidget(self.loadings_plot, *(1, 1), 1, 1)
        self.grid_container.addWidget(self.explained_variance_plot,
                                      *(2, 0), 1, 1)
        self.grid_container.addLayout(self.reconstructed_spectra_layout,
                                      *(2, 1), 1, 1)

    def connect_event_handlers(self):
        self.first_pc_scores_combo.currentIndexChanged.connect(
            self.update_plots)
        self.second_pc_scores_combo.currentIndexChanged.connect(
            self.update_plots)
        self.pc_loadings_combo.currentIndexChanged.connect(self.update_plots)

        self.x_coord_combo.currentIndexChanged.connect(self.update_plots)
        self.y_coord_combo.currentIndexChanged.connect(self.update_plots)
        self.z_coord_combo.currentIndexChanged.connect(self.update_plots)
        self.sample_combo.currentIndexChanged.connect(self.update_plots)

        self.pcs_for_reconstruction_combo.currentIndexChanged.connect(
            self.update_plots)

    def update_data(self, pca_data):
        self.input_datatype = type(pca_data)
        self.input_data = pca_data
        # if type(pca_data) in [spectroscopy_data, raman_image]:
        #     self.pcr_object = self.input_data.pca
        # else:  # is assumed to be instance of principal_component_regression
        self.pcr_object = self.input_data

        self.init_combo_boxes()
        self.update_plots()

    def init_combo_boxes(self):
        # Disables the update_spectra_plots function.
        self.plots_active = False

        self.first_pc_scores_combo.clear()
        self.first_pc_scores_combo.addItems(
            self.pcr_object.pca_scores.columns.to_numpy().astype(str))
        self.first_pc_scores_combo.setCurrentIndex(0)

        self.second_pc_scores_combo.clear()
        self.second_pc_scores_combo.addItems(
            self.pcr_object.pca_scores.columns.to_numpy().astype(str))
        self.second_pc_scores_combo.setCurrentIndex(1)

        self.pc_loadings_combo.clear()
        self.pc_loadings_combo.addItems(
            self.pcr_object.pca_loadings.columns.to_numpy().astype(str))
        self.pc_loadings_combo.setCurrentIndex(0)

        self.sample_combo.clear()
        self.x_coord_combo.clear()
        self.y_coord_combo.clear()
        self.z_coord_combo.clear()

        if self.input_datatype is raman_image:
            self.x_coord_combo.addItems(
                np.char.mod('%s',np.around(
                    self.input_data.get_coord_values(
                        'real',axis = 'x'),1)))

            self.y_coord_combo.addItems(
                np.char.mod('%s',np.around(
                    self.input_data.get_coord_values(
                        'real',axis = 'y'),1)))

            self.z_coord_combo.addItems(
                np.char.mod('%s',np.around(
                    self.input_data.get_coord_values(
                        'real',axis = 'z'),1)))
            
            if len(self.input_data.get_coord_values('coded',axis = 'x')) > 1:
                self.x_coord_combo.setEnabled(True)
            else:
                self.x_coord_combo.setEnabled(False)
            if len(self.input_data.get_coord_values('coded',axis = 'y')) > 1:
                self.y_coord_combo.setEnabled(True)
            else:
                self.y_coord_combo.setEnabled(False)
            if len(self.input_data.get_coord_values('coded',axis = 'z')) > 1:
                self.z_coord_combo.setEnabled(True)
            else:
                self.z_coord_combo.setEnabled(False)

            self.sample_combo.setEnabled(False)
        else:
            self.x_coord_combo.setEnabled(False)
            self.y_coord_combo.setEnabled(False)
            self.z_coord_combo.setEnabled(False)
            self.sample_combo.setEnabled(True)
            self.sample_combo.addItems(
                self.pcr_object.pca_scores.index.values.astype(str))
            self.sample_combo.setCurrentIndex(0)

        self.pcs_for_reconstruction_combo.clear()
        self.pcs_for_reconstruction_combo.addItems(
            self.pcr_object.pca_scores.columns.to_numpy().astype(str))
        self.pcs_for_reconstruction_combo.setCurrentIndex(0)

        # Now the update_spectra_plots function will do something.
        self.plots_active = True

    def update_plots(self):
        if self.plots_active is False:
            return

        self.scores_plot.axes.clear()
        self.loadings_plot.axes.clear()
        self.explained_variance_plot.axes.clear()

        self.scores_plot.plot(
            self.pcr_object.pca_scores[
                int(self.first_pc_scores_combo.currentText())],
            self.pcr_object.pca_scores[
                int(self.second_pc_scores_combo.currentText())],
            pen='b', mode='scatter')
        # self.scores_plot.axes.axhline(0, ls='dotted')
        # self.scores_plot.axes.axvline(0, ls='dotted')

        self.loadings_plot.plot(
            self.pcr_object.pca_loadings.index,
            self.pcr_object.pca_loadings[
                int(self.pc_loadings_combo.currentText())],
            pen='b', mode='line')

        x_explained_variance = np.insert(
            self.pcr_object.pca_explained_variance.index.values, 0, 0)
        y_explained_variance = np.insert(
            self.pcr_object.pca_explained_variance['cum'].values, 0, 0)
        for ii, jj in zip(x_explained_variance[1:], y_explained_variance[1:]):
            self.explained_variance_plot.axes.annotate(
                str(round(jj, 3)), xy=(ii, 0.05))
        self.explained_variance_plot.plot(
            x_explained_variance, y_explained_variance, pen='b', mode='line')

        reconstruction_pca_components = int(
            self.pcs_for_reconstruction_combo.currentText())
        self.reconstructed_data = self.pcr_object.reconstruct_data(
            used_pcs=reconstruction_pca_components)

        if self.input_datatype is raman_image:
            selected_spectrum_index = (
                self.get_coord(
                    'coded',
                    coord_value=float(self.x_coord_combo.currentText())),
                self.get_coord(
                    'coded',
                    coord_value=float(self.y_coord_combo.currentText())),
                self.get_coord(
                    'coded',
                    coord_value=float(self.z_coord_combo.currentText())))
        else:
            selected_spectrum_index = self.sample_combo.currentText()

        self.reconstructed_spectra_plot.axes.clear()
        self.reconstructed_spectra_plot.plot(
            self.pcr_object.x.columns,
            self.pcr_object.reconstruct_input().loc[
                selected_spectrum_index, :],
            pen='k')
        self.reconstructed_spectra_plot.plot(
            self.reconstructed_data.columns,
            self.reconstructed_data.loc[
                selected_spectrum_index, :],
            pen='r')

    def center(self):  # centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def get_coord(self, value_sort, axis='x', coord_value=None,
                  mode='raw_data'):

        if coord_value is not None:
            if value_sort == 'coded':
                return_value = round(
                    coord_value*self.input_data.coord_conversion_factor)
            else:
                return_value = (coord_value/
                                self.input_data.coord_conversion_factor)
        else:
            if mode == 'raw_data':
                if axis == 'x':
                    return_value = float(self.x_coord_combo.currentText())
                elif axis == 'y':
                    return_value = float(self.y_coord_combo.currentText())
                elif axis == 'z':
                    return_value = float(self.z_coord_combo.currentText())
            elif mode == 'processed':
                if axis == 'x':
                    return_value = float(
                        self.x_coord_combo_edited.currentText())
                elif axis == 'y':
                    return_value = float(
                        self.y_coord_combo_edited.currentText())
                elif axis == 'z':
                    return_value = float(
                        self.z_coord_combo_edited.currentText())

            if value_sort == 'coded':
                return_value = round(
                    return_value*self.input_data.coord_conversion_factor)

        return return_value
