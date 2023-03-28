///////////////////////////////////////////////////////////////////////////////
///
///	\file    Variable.cpp
///	\author  Paul Ullrich
///	\version July 22, 2018
///
///	<remarks>
///		Copyright 2000-2018 Paul Ullrich
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

#include "Variable.h"
#include "STLStringHelper.h"
#include "Exception.h"

///////////////////////////////////////////////////////////////////////////////
// VariableRegistry
///////////////////////////////////////////////////////////////////////////////

VariableRegistry::VariableRegistry() {}

///////////////////////////////////////////////////////////////////////////////

VariableRegistry::~VariableRegistry() {
	for (int v = 0; v < m_vecVariables.size(); v++) {
		delete m_vecVariables[v];
	}
}

///////////////////////////////////////////////////////////////////////////////

void VariableRegistry::InsertUniqueOrDelete(
	Variable * pvar,
	VariableIndex * pvarix
) {
	for (int v = 0; v < m_vecVariables.size(); v++) {
		if (*pvar == *(m_vecVariables[v])) {
			delete pvar;
			if (pvarix != NULL) {
				*pvarix = v;
			}
			return;
		}
	}

	m_vecVariables.push_back(pvar);
	if (pvarix != NULL) {
		*pvarix = m_vecVariables.size()-1;
	}
}

///////////////////////////////////////////////////////////////////////////////

int VariableRegistry::FindOrRegisterSubStr(
	const std::string & strIn,
	VariableIndex * pvarix
) {
	// Does not exist
	Variable * pvar = new Variable();
	_ASSERT(pvar != NULL);

	if (pvarix != NULL) {
		*pvarix = InvalidVariableIndex;
	}

	// Parse the string
	bool fParamMode = false;
	bool fDimMode = false;
	std::string strDim;

	if (strIn.length() >= 1) {
		if (strIn[0] == '_') {
			pvar->m_fOp = true;
		}
	}

	for (int n = 0; n <= strIn.length(); n++) {

		// Reading the variable name
		if (!fDimMode) {
			if (n == strIn.length()) {
				if (fParamMode) {
					_EXCEPTIONT("Unbalanced curly brackets in variable");
				}
				pvar->m_strName = strIn;
				InsertUniqueOrDelete(pvar, pvarix);
				return n;
			}

			// Items in curly brackets are included in variable name
			if (fParamMode) {
				if (strIn[n] == '(') {
					_EXCEPTIONT("Unexpected \'(\' in variable");
				}
				if (strIn[n] == ')') {
					_EXCEPTIONT("Unexpected \')\' in variable");
				}
				if (strIn[n] == '{') {
					_EXCEPTIONT("Unexpected \'{\' in variable");
				}
				if (strIn[n] == '}') {
					fParamMode = false;
				}
				continue;
			}
			if (strIn[n] == '{') {
				fParamMode = true;
				continue;
			}

			if (strIn[n] == ',') {
				pvar->m_strName = strIn.substr(0, n);
				InsertUniqueOrDelete(pvar, pvarix);
				return n;
			}
			if (strIn[n] == '(') {
				pvar->m_strName = strIn.substr(0, n);
				fDimMode = true;
				continue;
			}
			if (strIn[n] == ')') {
				pvar->m_strName = strIn.substr(0, n);
				InsertUniqueOrDelete(pvar, pvarix);
				return n;
			}

		// Reading in dimensions
		} else if (!pvar->m_fOp) {
			if (pvar->m_strArg.size() >= MaxVariableArguments) {
				_EXCEPTION1("Sanity check fail: Only %i dimensions / arguments may "
					"be specified", MaxVariableArguments);
			}
			if (n == strIn.length()) {
				_EXCEPTION1("Variable dimension list must be terminated"
					" with ): %s", strIn.c_str());
			}
			if ((strIn[n] == ',') || (strIn[n] == ')')) {
				if (strDim.length() == 0) {
					_EXCEPTIONT("Invalid dimension index in variable");
				}
				pvar->m_strArg.push_back(strDim);
				pvar->m_varArg.push_back(InvalidVariableIndex);
				if (strDim == ":") {
					pvar->m_fFreeArg.push_back(true);
				} else {
					pvar->m_fFreeArg.push_back(false);
				}
				strDim = "";

				if (strIn[n] == ')') {
					InsertUniqueOrDelete(pvar, pvarix);
					return (n+1);
				}

			} else {
				strDim += strIn[n];
			}

		// Reading in arguments
		} else {
			if (pvar->m_strArg.size() >= MaxVariableArguments) {
				_EXCEPTION1("Sanity check fail: Only %i dimensions / arguments may "
					"be specified", MaxVariableArguments);
			}
			if (n == strIn.length()) {
				_EXCEPTION1("Op argument list must be terminated"
					" with ): %s", strIn.c_str());
			}

			// No arguments
			if (strIn[n] == ')') {
				InsertUniqueOrDelete(pvar, pvarix);
				return (n+1);
			}

			// Check for floating point argument
			if (isdigit(strIn[n]) || (strIn[n] == '.') || (strIn[n] == '-')) {
				int nStart = n;
				for (; n <= strIn.length(); n++) {
					if (n == strIn.length()) {
						_EXCEPTION1("Op argument list must be terminated"
							" with ): %s", strIn.c_str());
					}
					if ((strIn[n] == ',') || (strIn[n] == ')')) {
						break;
					}
				}

				std::string strFloat = strIn.substr(nStart, n-nStart);
				if (!STLStringHelper::IsFloat(strFloat)) {
					_EXCEPTION2("Invalid floating point number at position %i in: %s",
						nStart, strIn.c_str());
				}
				pvar->m_strArg.push_back(strFloat);
				pvar->m_varArg.push_back(InvalidVariableIndex);
				pvar->m_fFreeArg.push_back(false);

			// Check for string argument
			} else if (strIn[n] == '\"') {
				int nStart = n;
				for (; n <= strIn.length(); n++) {
					if (n == strIn.length()) {
						_EXCEPTION1("String must be terminated with \": %s",
							strIn.c_str());
					}
					if (strIn[n] == '\"') {
						break;
					}
				}
				if (n >= strIn.length()-1) {
					_EXCEPTION1("Op argument list must be terminated"
						" with ): %s", strIn.c_str());
				}
				if ((strIn[n+1] != ',') && (strIn[n+1] != ')')) {
					_EXCEPTION2("Invalid character in argument list after "
						"string at position %i in: %s",
						n+1, strIn.c_str());
				}

				pvar->m_strArg.push_back(strIn.substr(nStart+1,n-nStart-1));
				pvar->m_varArg.push_back(InvalidVariableIndex);
				pvar->m_fFreeArg.push_back(false);

			// Check for variable
			} else {
				VariableIndex varix;
				n += FindOrRegisterSubStr(strIn.substr(n), &varix);

				pvar->m_strArg.push_back("");
				pvar->m_varArg.push_back(varix);
				pvar->m_fFreeArg.push_back(false);
			}

			if (strIn[n] == ')') {
				InsertUniqueOrDelete(pvar, pvarix);
				return (n+1);
			}
		}
	}

	_EXCEPTION1("Malformed variable string \"%s\"", strIn.c_str());
}

///////////////////////////////////////////////////////////////////////////////

Variable & VariableRegistry::Get(
	VariableIndex varix
) {
	if ((varix < 0) || (varix >= m_vecVariables.size())) {
		_EXCEPTION1("Variable index (%i) out of range", varix);
	}
	return *(m_vecVariables[varix]);
}

///////////////////////////////////////////////////////////////////////////////

std::string VariableRegistry::GetVariableString(
	VariableIndex varix
) const {
	if ((varix < 0) || (varix >= m_vecVariables.size())) {
		_EXCEPTION1("Variable index (%i) out of range", varix);
	}
	return m_vecVariables[varix]->ToString(*this);
}

///////////////////////////////////////////////////////////////////////////////
// Variable
///////////////////////////////////////////////////////////////////////////////

bool Variable::operator==(
	const Variable & var
) {
	if (m_fOp != var.m_fOp) {
		return false;
	}
	if (m_strName != var.m_strName) {
		return false;
	}
	if (m_strArg.size() != var.m_strArg.size()) {
		return false;
	}
	_ASSERT(m_strArg.size() == m_varArg.size());
	_ASSERT(m_strArg.size() == var.m_varArg.size());
	for (int i = 0; i < m_strArg.size(); i++) {
		if (m_strArg[i] != var.m_strArg[i]) {
			return false;
		}
		if (m_varArg[i] != var.m_varArg[i]) {
			return false;
		}
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////

std::string Variable::ToString(
	const VariableRegistry & varreg
) const {
	_ASSERT(m_varArg.size() == m_strArg.size());
	char szBuffer[20];
	std::string strOut = m_strName;
	if (m_varArg.size() != 0) {
		strOut += "(";
		for (size_t d = 0; d < m_varArg.size(); d++) {
			if (m_varArg[d] != InvalidVariableIndex) {
				strOut += varreg.GetVariableString(m_varArg[d]);
			} else {
				strOut += m_strArg[d];
			}
			if (d != m_varArg.size()-1) {
				strOut += ",";
			} else {
				strOut += ")";
			}
		}
	}
	return strOut;
}
