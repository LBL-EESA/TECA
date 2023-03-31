///////////////////////////////////////////////////////////////////////////////
///
///	\file    Variable.h
///	\author  Paul Ullrich
///	\version November 18, 2015
///
///	<remarks>
///		Copyright 2000-2014 Paul Ullrich
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
///   Removed some classes (DimInfo, DimInfoVector, VariableAuxIndexIterator),
///   some functions (FindOrRegister, UnloadAllGridData,
///   GetDependentVariableIndicesRecurse, GetDependentVariableIndices,
///   GetDependentVariableNames, GetAuxiliaryDimInfo,
///   GetAuxiliaryDimInfoAndVerifyConsistency, GetAuxiliaryDimInfo,
///   AssignAuxiliaryIndicesRecursive, ClearProcessingQueue,
///   AppendVariableToProcessingQueue, GetProcessingQueueVarPos,
///   GetProcessingQueueVarIx, GetProcessingQueueVariable,
///   GetProcessingQueueAuxIx, GetProcessingQueueAuxSize,
///   GetProcessingQueueOffset, AdvanceProcessingQueue, ResetProcessingQueue,
///   GetDataOp, GetArgumentCount, GetArgumentStrings, GetArgumentVarIxs,
///   GetFillValueFloat, GetNcVarFromNcFileVector, LoadGridData,
///   UnloadGridData, GetData), some variables (m_domDataOp,
///   m_sProcessingQueueVarPos, m_vecProcessingQueue, m_dFillValueFloat,
///   m_strUnits, m_vecAuxDimNames, m_iTimeDimIx, m_iVerticalDimIx,
///   m_nVerticalDimOrder, m_fNoTimeInNcFile, m_timeStored, m_data), and
///   some typedef (VariableDimIndex, VariableAuxIndex, DataMap).

#ifndef _VARIABLE_H_
#define _VARIABLE_H_

#include <vector>
#include <string>

///////////////////////////////////////////////////////////////////////////////

class Variable;

///	<summary>
///		A vector of pointers to Variable.
///	</summary>
typedef std::vector<Variable *> VariableVector;

///	<summary>
///		A unique index assigned to each Variable.
///	</summary>
typedef int VariableIndex;

///	<summary>
///		The invalid variable index.
///	</summary>
static const VariableIndex InvalidVariableIndex = (-1);

///	<summary>
///		A vector for storing VariableIndex.
///	</summary>
class VariableIndexVector : public std::vector<VariableIndex> {};

///////////////////////////////////////////////////////////////////////////////

class VariableRegistry {

public:
	///	<summary>
	///		Maximum number of arguments in variable.
	///	</summary>
	static const int MaxVariableArguments = 10;

public:
	///	<summary>
	///		Constructor (build an empty registry).
	///	</summary>
	VariableRegistry();

	///	<summary>
	///		Destructor.
	///	</summary>
	~VariableRegistry();

protected:
	///	<summary>
	///		Find this Variable in the registry.  If it does exist then
	///		return the corresponding VariableIndex and delete pvar.  If it
	///		does not exist then insert it into the registry.
	///	</summary>
	void InsertUniqueOrDelete(
		Variable * pvar,
		VariableIndex * pvarix
	);

public:
	///	<summary>
	///		Parse the given recursively string and register all relevant
	///		Variables.  At the end of the Variable definition return
	///		the current position in the string.
	///	</summary>
	int FindOrRegisterSubStr(
		const std::string & strIn,
		VariableIndex * pvarix
	);

	///	<summary>
	///		Get the Variable with the specified index.
	///	</summary>
	Variable & Get(VariableIndex varix);

	///	<summary>
	///		Get the descriptor for the Variable with the specified index.
	///	</summary>
	std::string GetVariableString(VariableIndex varix) const;

private:
	///	<summary>
	///		Array of variables.
	///	</summary>
	VariableVector m_vecVariables;
};

///////////////////////////////////////////////////////////////////////////////

///	<summary>
///		A class for storing a 2D slice of data, and associated metadata.
///	</summary>
class Variable {

friend class VariableRegistry;

public:
	///	<summary>
	///		Default constructor.
	///	</summary>
	Variable() :
		m_strName(),
		m_fOp(false)
	{ }

public:
	///	<summary>
	///		Equality operator.
	///	</summary>
	bool operator==(const Variable & var);

	///	<summary>
	///		Get the name of this Variable.
	///	</summary>
	const std::string & GetName() const {
		return m_strName;
	}

	///	<summary>
	///		Check if this Variable is an operator.
	///	</summary>
	bool IsOp() const {
		return m_fOp;
	}

public:
	///	<summary>
	///		Get a string representation of this variable.
	///	</summary>
	std::string ToString(
		const VariableRegistry & varreg
	) const;

protected:
	///	<summary>
	///		Variable name.
	///	</summary>
	std::string m_strName;

	///	<summary>
	///		Flag indicating this is an operator.
	///	</summary>
	bool m_fOp;

	///	<summary>
	///		Free operator arguments.
	///	</summary>
	std::vector<bool> m_fFreeArg;

	///	<summary>
	///		Specified operator arguments or auxiliary indices (as std::string).
	///	</summary>
	std::vector<std::string> m_strArg;

	///	<summary>
	///		Specified operator arguments (as VariableIndex).
	///	</summary>
	VariableIndexVector m_varArg;
};

///////////////////////////////////////////////////////////////////////////////

#endif

