import json
import math
from flask import Flask, jsonify, request, send_from_directory
import trueskill
from itertools import combinations
from datetime import datetime, timezone
import os

# --- Configuration ---
# For security, it's better to get the secret from an environment variable
ADMIN_SECRET = os.environ.get("VOLLEY_ADMIN_SECRET", "change-me-to-a-real-password")
DATA_FILE_PATH = "volley-pro-matches.json"
# Set a fixed start date for the season (September 1, 2025)
SEASON_START = datetime(2025, 9, 1, tzinfo=timezone.utc)

# --- TrueSkill Setup ---
trueskill.setup(mu=25.0, sigma=25.0 / 3.0, beta=25.0 / 6.0, tau=25.0 / 300.0, draw_probability=0)
ts_env = trueskill.TrueSkill()


def _win_probability(team1_ratings, team2_ratings):
    """Calculates the win probability for team 1 against team 2."""
    delta_mu = sum(r.mu for r in team1_ratings) - sum(r.mu for r in team2_ratings)
    sum_sigma_sq = sum(r.sigma ** 2 for r in team1_ratings) + sum(r.sigma ** 2 for r in team2_ratings)
    beta_sq = ts_env.beta ** 2
    denom = math.sqrt(len(team1_ratings) * beta_sq + len(team2_ratings) * beta_sq + sum_sigma_sq)
    if denom == 0: return 0.5
    return 0.5 * (1 + math.erf(delta_mu / (math.sqrt(2) * denom)))


app = Flask(__name__, static_folder='.', static_url_path='')


def load_data():
    """Loads match and player data from the JSON file."""
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return a default structure if the file doesn't exist or is empty
        return {"matches": [], "playerDetails": {}}


def save_data(data):
    """Saves the given data to the JSON file."""
    with open(DATA_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def calculate_all_stats():
    """
    The main calculation engine. Processes all matches to generate stats,
    leaderboards, history, and other analytics. This function is now responsible
    for all data processing, moving the load from the client to the server.
    """
    data = load_data()
    matches = data.get("matches", [])
    player_details = data.get("playerDetails", {})

    # Determine week number for each match based on a fixed season start date
    if matches:
        season_start_ts = SEASON_START.timestamp() * 1000
        ONE_WEEK_MS = 7 * 24 * 60 * 60 * 1000
        for match in matches:
            if isinstance(match.get('ts'), (int, float)):
                time_diff = match['ts'] - season_start_ts
                match['week'] = math.floor(time_diff / ONE_WEEK_MS) + 1 if time_diff >= 0 else 0
            else:
                match['week'] = 1  # Fallback for old matches without a timestamp

    player_names = set(p for m in matches for p in m['teamA'] + m['teamB'])
    if not player_names:
        return {"leaderboard": [], "playerStats": {}, "matches": [], "playerDetails": {}, "history": {},
                "superlatives": {}, "rivalries": [], "duos": [], "totalMatches": 0, "weeklyData": {}}

    # Initialize ratings, history, and superlatives
    ratings = {name: ts_env.create_rating() for name in player_names}
    history = {name: [{'mu': r.mu, 'sigma': r.sigma, 'match_id': -1}] for name, r in ratings.items()}
    last_played_index = {name: -1 for name in player_names}
    superlatives = {
        'upset': {'prob': 1},
        'dominant': {'margin': 0},
        'best_performance': {'skill_gain': 0},
        'highCaliber': {'avgSkill': 0}
    }
    rivalries, duos = {}, {}
    weekly_data_aggregator = {}

    # --- Process every match sequentially ---
    for match_index, match in enumerate(matches):
        team_a_ratings = {p: ratings[p] for p in match['teamA']}
        team_b_ratings = {p: ratings[p] for p in match['teamB']}

        # Pre-match calculations for superlatives
        win_prob_a = _win_probability(list(team_a_ratings.values()), list(team_b_ratings.values()))
        winner_prob = win_prob_a if match['scoreA'] > match['scoreB'] else 1 - win_prob_a
        if winner_prob < superlatives['upset']['prob']:
            superlatives['upset'] = {'prob': winner_prob, 'match_id': match_index}

        margin = abs(match['scoreA'] - match['scoreB'])
        if margin > superlatives['dominant']['margin']:
            superlatives['dominant'] = {'margin': margin, 'match_id': match_index}

        avg_skill_a = sum(r.mu - 3 * r.sigma for r in team_a_ratings.values()) / len(
            team_a_ratings) if team_a_ratings else 0
        avg_skill_b = sum(r.mu - 3 * r.sigma for r in team_b_ratings.values()) / len(
            team_b_ratings) if team_b_ratings else 0
        avg_match_skill = (avg_skill_a + avg_skill_b) / 2
        if avg_match_skill > superlatives['highCaliber']['avgSkill']:
            superlatives['highCaliber'] = {'avgSkill': avg_match_skill, 'match_id': match_index}

        # Update duo and rivalry stats
        for team, won in [(match['teamA'], match['scoreA'] > match['scoreB']),
                          (match['teamB'], match['scoreB'] > match['scoreA'])]:
            for p1, p2 in combinations(team, 2):
                key = tuple(sorted((p1, p2)))
                duos.setdefault(key, {'wins': 0, 'total': 0, 'players': key})['total'] += 1
                if won: duos[key]['wins'] += 1
        for p1 in match['teamA']:
            for p2 in match['teamB']:
                key = tuple(sorted((p1, p2)))
                rivalries.setdefault(key, {'p1_wins': 0, 'p2_wins': 0, 'total': 0, 'players': key})['total'] += 1
                if match['scoreA'] > match['scoreB']:
                    rivalries[key]['p1_wins' if p1 == key[0] else 'p2_wins'] += 1
                else:
                    rivalries[key]['p2_wins' if p1 == key[0] else 'p1_wins'] += 1

        # --- Update TrueSkill ratings ---
        ranks = [0, 1] if match['scoreA'] > match['scoreB'] else [1, 0]
        new_ratings_map = ts_env.rate([team_a_ratings, team_b_ratings], ranks=ranks)
        new_ratings = {p: r for team in new_ratings_map for p, r in team.items()}

        week_num = match.get('week', 0)
        if week_num > 0:
            weekly_data_aggregator.setdefault(week_num, {'skillGains': {}, 'matchesPlayed': {}})

        # Update history and track performance for each player in the match
        for player_name, new_rating in new_ratings.items():
            old_rating = ratings[player_name]
            skill_gain = (new_rating.mu - 3 * new_rating.sigma) - (old_rating.mu - 3 * old_rating.sigma)
            if skill_gain > superlatives['best_performance']['skill_gain']:
                superlatives['best_performance'] = {'skill_gain': skill_gain, 'player': player_name,
                                                    'match_id': match_index}

            if week_num > 0:
                weekly_data_aggregator[week_num]['skillGains'][player_name] = weekly_data_aggregator[week_num][
                                                                                  'skillGains'].get(player_name,
                                                                                                    0) + skill_gain
                weekly_data_aggregator[week_num]['matchesPlayed'][player_name] = weekly_data_aggregator[week_num][
                                                                                     'matchesPlayed'].get(player_name,
                                                                                                          0) + 1

            ratings[player_name] = new_rating
            history[player_name].append({'mu': new_rating.mu, 'sigma': new_rating.sigma, 'match_id': match_index})

        for p in match['teamA'] + match['teamB']: last_played_index[p] = match_index

        if week_num > 0:
            current_leaderboard = sorted(
                [{'name': name, 'skill': r.mu - 3 * r.sigma} for name, r in ratings.items()],
                key=lambda x: x['skill'], reverse=True
            )
            weekly_data_aggregator[week_num]['end_of_week_leaderboard_snapshot'] = current_leaderboard

    # --- Post-Calculation Analysis ---
    final_leaderboard = sorted(
        [{'name': name, 'skill': r.mu - 3 * r.sigma} for name, r in ratings.items()],
        key=lambda x: x['skill'], reverse=True
    )

    # Calculate detailed stats for each player
    player_stats = {}
    for name in player_names:
        p_history = history[name]
        skill_history = [h['mu'] - 3 * h['sigma'] for h in p_history]

        wins, losses, win_streak, longest_streak = 0, 0, 0, 0
        clutch_wins, clutch_attempts = 0, 0
        current_streak, streak_type = 0, ''

        # Streaks
        if len(p_history) > 1:
            streak_type = 'W' if p_history[-1]['mu'] > p_history[-2]['mu'] else 'L'
            for i in range(len(p_history) - 1, 0, -1):
                is_win = p_history[i]['mu'] > p_history[i - 1]['mu']
                if (is_win and streak_type == 'W') or (not is_win and streak_type == 'L'):
                    current_streak += 1
                else:
                    break

        # Wins, losses, clutch
        for m in matches:
            is_player_in_match = name in m['teamA'] or name in m['teamB']
            if is_player_in_match:
                won = (name in m['teamA'] and m['scoreA'] > m['scoreB']) or \
                      (name in m['teamB'] and m['scoreB'] > m['scoreA'])
                if won:
                    wins += 1;
                    win_streak += 1
                else:
                    losses += 1;
                    win_streak = 0
                longest_streak = max(longest_streak, win_streak)

                is_clutch = abs(m['scoreA'] - m['scoreB']) == 1
                if is_clutch:
                    clutch_attempts += 1
                    if won: clutch_wins += 1

        total_games = wins + losses
        win_rate = (wins / total_games * 100) if total_games > 0 else 0
        clutch_rate = (clutch_wins / clutch_attempts * 100) if clutch_attempts > 0 else 0

        # Skill change over last 10 games
        if len(skill_history) > 10:
            skill_change_10 = skill_history[-1] - skill_history[-11]
        elif len(skill_history) > 1:
            skill_change_10 = skill_history[-1] - skill_history[0]
        else:
            skill_change_10 = 0

        player_stats[name] = {
            "name": name, "wins": wins, "losses": losses, "total": total_games,
            "winRate": f"{win_rate:.1f}", "skill": ratings[name].mu - 3 * ratings[name].sigma,
            "peakSkill": max(skill_history) if skill_history else 0,
            "mu": ratings[name].mu, "sigma": ratings[name].sigma,
            "longestWinStreak": longest_streak,
            "currentStreak": {'count': current_streak, 'type': streak_type},
            "clutchRate": clutch_rate,
            "skillChangeLast10": skill_change_10,
            "last_played": last_played_index[name]
        }

    # Finalize weekly data
    final_weekly_data = {}
    for week_num, week_data in weekly_data_aggregator.items():
        top_performer = max(week_data['skillGains'], key=week_data['skillGains'].get, default=None)
        most_active = max(week_data['matchesPlayed'], key=week_data['matchesPlayed'].get, default=None)

        # FIX: Filter the weekly leaderboard to only include players active that week for relevance
        active_players_this_week = set(week_data['matchesPlayed'].keys())
        full_leaderboard = week_data.get('end_of_week_leaderboard_snapshot', [])
        filtered_leaderboard = [p for p in full_leaderboard if p['name'] in active_players_this_week]

        final_weekly_data[week_num] = {
            "weekNumber": week_num,
            "leaderboard": filtered_leaderboard,
            "topPerformer": {"name": top_performer,
                             "value": week_data['skillGains'].get(top_performer, 0)} if top_performer else None,
            "mostActive": {"name": most_active,
                           "value": week_data['matchesPlayed'].get(most_active, 0)} if most_active else None
        }

    # Filter and sort duos and rivalries
    final_duos = sorted([d for d in duos.values() if d['total'] >= 5],
                        key=lambda x: (x['wins'] / x['total'], x['total']), reverse=True)[:10]
    final_rivalries = sorted([r for r in rivalries.values() if r['total'] >= 5], key=lambda x: x['total'],
                             reverse=True)[:10]

    return {
        "leaderboard": final_leaderboard, "playerStats": player_stats, "matches": matches,
        "playerDetails": player_details, "history": history,
        "superlatives": superlatives, "rivalries": final_rivalries, "duos": final_duos,
        "totalMatches": len(matches), "weeklyData": final_weekly_data
    }


# --- API Routes ---
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


@app.route('/api/data')
def get_all_data():
    return jsonify(calculate_all_stats())


@app.route('/api/add_match', methods=['POST'])
def add_match():
    payload = request.json
    if payload.get('secret') != ADMIN_SECRET:
        return jsonify({"error": "Invalid admin secret"}), 403
    data = load_data()
    data['matches'].append({k: payload[k] for k in ['teamA', 'teamB', 'scoreA', 'scoreB', 'ts']})
    # Add new players to details if they don't exist
    for player in payload['teamA'] + payload['teamB']:
        if player not in data.get('playerDetails', {}):
            data.setdefault('playerDetails', {})[player] = {"lastName": "", "abbrev": "", "number": ""}
    save_data(data)
    return jsonify({"success": True})


@app.route('/api/update_player', methods=['POST'])
def update_player():
    payload = request.json
    if payload.get('secret') != ADMIN_SECRET:
        return jsonify({"error": "Invalid admin secret"}), 403

    old_name, new_name, details = payload['originalName'], payload['newName'], payload['details']
    data = load_data()

    if old_name != new_name:
        # Check for name collision
        if new_name in data.get('playerDetails', {}):
            return jsonify({"error": f"Player '{new_name}' already exists."}), 409
        # Rename player in all matches
        for match in data.get('matches', []):
            match['teamA'] = [new_name if p == old_name else p for p in match['teamA']]
            match['teamB'] = [new_name if p == old_name else p for p in match['teamB']]
        # Rename player in details
        if old_name in data.get('playerDetails', {}):
            data['playerDetails'][new_name] = data['playerDetails'].pop(old_name)

    data.setdefault('playerDetails', {})[new_name] = details
    save_data(data)
    return jsonify({"success": True})


if __name__ == '__main__':
    app.run(debug=True, port=5001)

