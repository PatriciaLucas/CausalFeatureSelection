# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 21:59:51 2023

@author: Patricia
"""

import sqlite3
import contextlib

def execute_insert(sql,data,database_path):
    """
    Função para executar INSERT INTO
    :parametro sql: string com código sql
    :parametro data: dados que serão inseridos no banco de dados
    :parametro database_path: caminho para o banco de dados
    """
    with contextlib.closing(sqlite3.connect(database_path)) as conn: # auto-closes
        with conn: # auto-commits
            with contextlib.closing(conn.cursor()) as cursor: # auto-closes
                cursor.execute(sql,data)
                return cursor.fetchall()


def execute(sql,database_path):
    """
    Função para executar INSERT INTO
    :parametro sql: string com código sql
    :parametro database_path: caminho para o banco de dados
    :return: dataframe com os valores retornados pela consulta sql
    """
    with contextlib.closing(sqlite3.connect(database_path)) as conn: # auto-closes
        with conn: # auto-commits
            with contextlib.closing(conn.cursor()) as cursor: # auto-closes
                cursor.execute(sql)
                return cursor.fetchall()
            
            
            
            
            
            