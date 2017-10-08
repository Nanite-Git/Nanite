# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 08:54:49 2017

@author: -holgDer
"""
import configparser
import mysql.connector
from mysql.connector import errorcode
from datetime import date
import time 

User = ''
Password = ''
Host = ''
Database = ''
Port = ''
Socket = ''

def my_configparser():
    config = configparser.ConfigParser()
    #config.read('MyConnection.INI')
    #print(config['MYSQL']['Host'])     # -> "/path/name/"
    
    User = "root"#config['MYSQL']['User']
    Password = ""#config['MYSQL']['Password']
    Host = "127.0.0.1"#config['MYSQL']['Host']
    Database = "test"#config['MYSQL']['Database']
    Port = "3306"#config['MYSQL']['Port']
    Socket = "0"#config['MYSQL']['Socket']

    print(User,Password,Host,Database,Port,Socket)
    cnx = connect_db(User, Password, Host, Database, Port, Socket)
    
    return config, cnx



def connect_db(User, Psw, Host = '127.0.0.1', Db = 'test', Port = 3306, Socket = ''):
    try:
      cnx = mysql.connector.connect(user=User,password=Psw,
                                  database = Db)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
            return -1
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
            return -1
        else:
            print(err)
            return -1
    else:
        print("connection to database established...")
        return cnx



def write_to_sql(text1, cnx_cursor): 
    
    cnx_cursor.execute(text1)
    return cnx_cursor.lastrowid



def read_flag_and_reset():
    dataset=""
    read_query = "SELECT flag FROM test.new_bar_flag"
    clearQuery = "TRUNCATE test.new_bar_flag;"
    write_query = "INSERT INTO test.new_bar_flag (flag) VALUES (0);"
    
    dataset = read_from_sql(cnx_cursor, read_query)
    if(dataset[0][0] == 1):
        print("getData...")
        write_to_sql(clearQuery, cnx_cursor)
        write_to_sql(write_query, cnx_cursor)

    return dataset[0][0]



def read_from_sql(cursor, query):
    dataset=[]
    row = ""
    #cnx_cursor.execute("USE ",table)
    cursor.execute(query)
    
    while row is not None:
        row = cursor.fetchone()
        dataset.append(row)
    
    return dataset