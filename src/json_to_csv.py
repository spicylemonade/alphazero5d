import csv

# Write to CSV
def to_csv(data):
    # Flatten the nested structure
    flattened_data = []

    for move in data:
     flattened_move = {
      'promotion': move['promotion'],
      'enPassant': move['enPassant'],
      'castling': move['castling'],
      'start_timeline': move['start']['timeline'],
      'start_turn': move['start']['turn'],
      'start_player': move['start']['player'],
      'start_coordinate': move['start']['coordinate'],
      'start_rank': move['start']['rank'],
      'start_file': move['start']['file'],
      'end_timeline': move['end']['timeline'],
      'end_turn': move['end']['turn'],
      'end_player': move['end']['player'],
      'end_coordinate': move['end']['coordinate'],
      'end_rank': move['end']['rank'],
      'end_file': move['end']['file'],
      'player': move['player'],
      'realEnd_timeline': move['realEnd']['timeline'],
      'realEnd_turn': move['realEnd']['turn'],
      'realEnd_player': move['realEnd']['player'],
      'realEnd_coordinate': move['realEnd']['coordinate'],
      'realEnd_rank': move['realEnd']['rank'],
      'realEnd_file': move['realEnd']['file']
     }
     flattened_data.append(flattened_move)
    with open('../chess_moves.csv', 'w', newline='') as csvfile:
        fieldnames = flattened_data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in flattened_data:
            writer.writerow(row)

    print("CSV file 'chess_moves.csv' has been created successfully.")

