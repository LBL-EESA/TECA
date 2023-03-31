///////////////////////////////////////////////////////////////////////////////
///
///	\file    SimpleGrid.h
///	\author  Paul Ullrich
///	\version August 30, 2019
///
///	<remarks>
///		Copyright 2019 Paul Ullrich
///
///		This file is distributed as part of the Tempest source code package.
///		Permission is granted to use, copy, modify and distribute this
///		source code and its documentation under the terms of the GNU General
///		Public License.  This software is provided "as is" without express
///		or implied warranty.
///	</remarks>
///
///   Amanda Dufek
///   March 28, 2023
///   Removed some functions (HasAreas, HasConnectivity,
///   GenerateLatitudeLongitude, GenerateRegionalLatitudeLongitude,
///   GenerateRectilinearStereographic, FromMeshFV, FromMeshFE, FromFile,
///   ToFile, FromUnstructuredDataFile, DimCount, CoordinateVectorToIndex,
///   BuildKDTree, NearestNode, NearestNodes)
///   and m_kdtree and c_szFileIdentifier variables.

#ifndef _SIMPLEGRID_H_
#define _SIMPLEGRID_H_

#include "DataArray1D.h"

#include <vector>

///////////////////////////////////////////////////////////////////////////////

///	<summary>
///		A data structure describing the grid, including coordinates of
///		each data point and graph connectivity of elements.
///	</summary>
class SimpleGrid {

public:
	///	<summary>
	///		Constructor.
	///	</summary>
	SimpleGrid() { }

	///	<summary>
	///		Destructor.
	///	</summary>
	~SimpleGrid();

	///	<summary>
	///		Determine if the SimpleGrid is initialized.
	///	</summary>
	bool IsInitialized() const;

public:
	///	<summary>
	///		Generate connectivity information for a rectilinear grid.
	///	</summary>
	void GenerateRectilinearConnectivity(
		int nLat,
		int nLon,
		bool fRegional,
		bool fDiagonalConnectivity
	);

	///	<summary>
	///		Generate the unstructured grid information for a
	///		longitude-latitude grid.
	///	</summary>
	void GenerateLatitudeLongitude(
		const DataArray1D<double> & vecLat,
		const DataArray1D<double> & vecLon,
		bool fRegional,
		bool fDiagonalConnectivity,
		bool fVerbose
	);

	///	<summary>
	///		Get the size of the SimpleGrid (number of points).
	///	</summary>
	size_t GetSize() const {
		_ASSERT(m_dLat.GetRows() == m_dLon.GetRows());
		return (m_dLon.GetRows());
	}

public:
	///	<summary>
	///		Grid dimensions.
	///	</summary>
	std::vector<size_t> m_nGridDim;

	///	<summary>
	///		Longitude of each grid point (in radians).
	///	</summary>
	DataArray1D<double> m_dLon;

	///	<summary>
	///		Latitude of each grid point (in radians).
	///	</summary>
	DataArray1D<double> m_dLat;

public:
	///	<summary>
	///		Area of the grid cell (in sr) (optionally initialized).
	///	</summary>
	DataArray1D<double> m_dArea;

	///	<summary>
	///		Connectivity of each grid point (optionally initialized).
	///	</summary>
	std::vector< std::vector<int> > m_vecConnectivity;
};

///////////////////////////////////////////////////////////////////////////////

#endif

