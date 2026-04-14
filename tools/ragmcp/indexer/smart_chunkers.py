#!/usr/bin/env python3
"""
Advanced chunking strategies for different file types.
Especially for PL/SQL packages - extract individual procedures/functions.
"""

import re
from typing import List, Dict, Any
from pathlib import Path


def chunk_plsql_by_function(content: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Smart chunking for PL/SQL packages.
    Extracts each PROCEDURE and FUNCTION as a separate chunk.

    This is CRITICAL for business logic - each function/procedure is a semantic unit!
    """
    chunks = []
    lines = content.split('\n')

    # Patterns to detect procedure/function boundaries
    proc_pattern = re.compile(r'^\s*(PROCEDURE|FUNCTION)\s+(\w+)', re.IGNORECASE)
    end_pattern = re.compile(r'^\s*END\s+(\w+)?\s*;', re.IGNORECASE)

    current_chunk = []
    in_procedure = False
    proc_name = None
    proc_type = None
    chunk_start_line = 1
    nesting_level = 0
    chunk_index = 0

    for i, line in enumerate(lines, 1):
        # Detect start of procedure/function
        match = proc_pattern.match(line)
        if match and not in_procedure:
            # Save previous chunk if exists
            if current_chunk and len(current_chunk) > 10:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'file_path': file_path,
                    'code_chunk': chunk_text,
                    'start_line': chunk_start_line,
                    'end_line': i - 1,
                    'file_type': 'plsql',
                    'chunk_type': 'package_header',
                    'function_name': None,
                    'chunk_index': chunk_index,
                    'total_chunks': -1
                })
                chunk_index += 1

            # Start new procedure/function chunk
            in_procedure = True
            proc_type = match.group(1).upper()
            proc_name = match.group(2)
            current_chunk = [line]
            chunk_start_line = i
            nesting_level = 1
            continue

        if in_procedure:
            current_chunk.append(line)

            # Track nesting level (BEGIN/END)
            if re.search(r'\bBEGIN\b', line, re.IGNORECASE):
                nesting_level += 1

            # Detect end of procedure/function
            end_match = end_pattern.match(line)
            if end_match:
                nesting_level -= 1

                # Check if this END matches our procedure
                end_name = end_match.group(1)
                if nesting_level == 0 or (end_name and end_name.upper() == proc_name.upper()):
                    # Complete procedure/function found!
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append({
                        'file_path': file_path,
                        'code_chunk': chunk_text,
                        'start_line': chunk_start_line,
                        'end_line': i,
                        'file_type': 'plsql',
                        'chunk_type': proc_type.lower(),
                        'function_name': proc_name,
                        'chunk_index': chunk_index,
                        'total_chunks': -1
                    })
                    chunk_index += 1

                    # Reset
                    current_chunk = []
                    in_procedure = False
                    proc_name = None
                    proc_type = None
                    chunk_start_line = i + 1
                    nesting_level = 0
        else:
            # Outside procedure - accumulate header/comments
            current_chunk.append(line)

    # Add final chunk (usually package end or remaining code)
    if current_chunk and len(current_chunk) > 10:
        chunk_text = '\n'.join(current_chunk)
        chunks.append({
            'file_path': file_path,
            'code_chunk': chunk_text,
            'start_line': chunk_start_line,
            'end_line': len(lines),
            'file_type': 'plsql',
            'chunk_type': 'package_footer',
            'function_name': None,
            'chunk_index': chunk_index,
            'total_chunks': -1
        })

    # Update total_chunks for all chunks
    for chunk in chunks:
        chunk['total_chunks'] = len(chunks)

    return chunks


def chunk_proc_by_function(content: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Smart chunking for Pro*C files.
    Extracts C functions with their EXEC SQL blocks.
    """
    chunks = []
    lines = content.split('\n')

    # Pattern to detect C function definitions
    func_pattern = re.compile(r'^\s*\w+\s+(\w+)\s*\([^)]*\)\s*{?\s*$')

    current_chunk = []
    in_function = False
    in_exec_sql = False
    func_name = None
    chunk_start_line = 1
    brace_depth = 0
    chunk_index = 0

    for i, line in enumerate(lines, 1):
        # Track EXEC SQL blocks
        if 'EXEC SQL' in line:
            in_exec_sql = True

        # Detect function start
        match = func_pattern.match(line)
        if match and not in_function:
            # Save previous chunk
            if current_chunk and len(current_chunk) > 10:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'file_path': file_path,
                    'code_chunk': chunk_text,
                    'start_line': chunk_start_line,
                    'end_line': i - 1,
                    'file_type': 'proc',
                    'chunk_type': 'header',
                    'function_name': None,
                    'chunk_index': chunk_index,
                    'total_chunks': -1
                })
                chunk_index += 1

            # Start new function
            in_function = True
            func_name = match.group(1)
            current_chunk = [line]
            chunk_start_line = i
            brace_depth = 0
            continue

        if in_function:
            current_chunk.append(line)

            # Track braces
            brace_depth += line.count('{') - line.count('}')

            # End of function?
            if brace_depth < 0 or (brace_depth == 0 and '}' in line and not in_exec_sql):
                # Function complete
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'file_path': file_path,
                    'code_chunk': chunk_text,
                    'start_line': chunk_start_line,
                    'end_line': i,
                    'file_type': 'proc',
                    'chunk_type': 'function',
                    'function_name': func_name,
                    'chunk_index': chunk_index,
                    'total_chunks': -1
                })
                chunk_index += 1

                # Reset
                current_chunk = []
                in_function = False
                func_name = None
                chunk_start_line = i + 1
                brace_depth = 0
        else:
            # Outside function - accumulate includes/defines
            current_chunk.append(line)

        # Track end of EXEC SQL
        if in_exec_sql and ';' in line:
            in_exec_sql = False

    # Add final chunk
    if current_chunk and len(current_chunk) > 10:
        chunk_text = '\n'.join(current_chunk)
        chunks.append({
            'file_path': file_path,
            'code_chunk': chunk_text,
            'start_line': chunk_start_line,
            'end_line': len(lines),
            'file_type': 'proc',
            'chunk_type': 'footer',
            'function_name': None,
            'chunk_index': chunk_index,
            'total_chunks': -1
        })

    # Update total_chunks
    for chunk in chunks:
        chunk['total_chunks'] = len(chunks)

    return chunks


def chunk_java_by_method(content: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Smart chunking for Java files.
    Extracts each method as a separate chunk.
    """
    chunks = []
    lines = content.split('\n')

    # Pattern for Java methods
    method_pattern = re.compile(
        r'^\s*(public|private|protected)?\s*(static)?\s*'
        r'(<[^>]+>\s*)?'  # Generics
        r'(\w+)\s+'       # Return type
        r'(\w+)\s*\([^)]*\)\s*'  # Method name and params
        r'(throws\s+[^{]+)?\s*{',
        re.IGNORECASE
    )

    current_chunk = []
    in_method = False
    method_name = None
    chunk_start_line = 1
    brace_depth = 0
    chunk_index = 0

    for i, line in enumerate(lines, 1):
        match = method_pattern.search(line)
        if match and not in_method:
            # Save previous chunk (class header, imports, etc.)
            if current_chunk and len(current_chunk) > 5:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'file_path': file_path,
                    'code_chunk': chunk_text,
                    'start_line': chunk_start_line,
                    'end_line': i - 1,
                    'file_type': 'java',
                    'chunk_type': 'class_header',
                    'function_name': None,
                    'chunk_index': chunk_index,
                    'total_chunks': -1
                })
                chunk_index += 1

            # Start new method
            in_method = True
            method_name = match.group(5)
            current_chunk = [line]
            chunk_start_line = i
            brace_depth = line.count('{') - line.count('}')
            continue

        if in_method:
            current_chunk.append(line)
            brace_depth += line.count('{') - line.count('}')

            # End of method?
            if brace_depth == 0:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'file_path': file_path,
                    'code_chunk': chunk_text,
                    'start_line': chunk_start_line,
                    'end_line': i,
                    'file_type': 'java',
                    'chunk_type': 'method',
                    'function_name': method_name,
                    'chunk_index': chunk_index,
                    'total_chunks': -1
                })
                chunk_index += 1

                # Reset
                current_chunk = []
                in_method = False
                method_name = None
                chunk_start_line = i + 1
        else:
            current_chunk.append(line)

    # Final chunk
    if current_chunk and len(current_chunk) > 5:
        chunk_text = '\n'.join(current_chunk)
        chunks.append({
            'file_path': file_path,
            'code_chunk': chunk_text,
            'start_line': chunk_start_line,
            'end_line': len(lines),
            'file_type': 'java',
            'chunk_type': 'class_footer',
            'function_name': None,
            'chunk_index': chunk_index,
            'total_chunks': -1
        })

    # Update total_chunks
    for chunk in chunks:
        chunk['total_chunks'] = len(chunks)

    return chunks


def get_smart_chunker(file_path: Path):
    """
    Returns the appropriate smart chunker based on file type.
    Returns None if simple chunking should be used.
    """
    suffix = file_path.suffix.lower()
    path_str = str(file_path).lower()

    # PL/SQL packages - ALWAYS use function-level chunking
    if '/pkg/' in path_str and suffix == '.sql':
        return chunk_plsql_by_function

    # Pro*C files - ALWAYS use function-level chunking
    if suffix == '.pc':
        return chunk_proc_by_function

    # Java files - use method-level chunking
    if suffix == '.java':
        return chunk_java_by_method

    # For other files, use simple line-based chunking
    return None


# Export functions
__all__ = [
    'chunk_plsql_by_function',
    'chunk_proc_by_function',
    'chunk_java_by_method',
    'get_smart_chunker'
]
