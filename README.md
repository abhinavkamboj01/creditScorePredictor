# DeFi Wallet Credit Scoring System

This project simulates a credit score system for DeFi wallets using transaction data (e.g., borrow, repay, deposit, liquidation). The model uses behavioral features to derive a score scaled from 0 to 1000.

## Features

- Extracts wallet behavior from JSON transaction logs
- Engineers risk-related financial features
- Simulates a credit scoring logic
- Normalizes raw score using MinMaxScaler
- CLI support for running on any dataset

## Architecture

- `io.py`: Loads/saves JSON and CSV files
- `processing.py`: Preprocesses and creates feature sets
- `scoring.py`: Applies scoring logic and normalization
- `main.py`: CLI entry to run the full pipeline

