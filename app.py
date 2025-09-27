import json
import math
from flask import Flask, jsonify, request, send_from_directory
import trueskill
from itertools import combinations
from datetime import datetime, timezone

# --- Configuration ---
ADMIN_SECRET = "change-me-to-a-real-password"
DATA_FILE_PATH = "volley-pro-matches.json"
INACTIVITY_THRESHOLD = 15
# NEW: Set a fixed start date for the season (September 1, 2025)
SEASON_START = datetime(2025, 9, 1, tzinfo=timezone.utc)

# --- TrueSkill Setup ---
trueskill.setup(mu=25.0, sigma=25.0 / 3.0, beta=25.0 / 6.0, tau=25.0 / 300.0, draw_probability=0)
ts_env = trueskill.TrueSkill()


def _win_probability(team1_ratings, team2_ratings):
    delta_mu = sum(r.mu for r in team1_ratings) - sum(r.mu for r in team2_ratings)
    sum_sigma_sq = sum(r.sigma ** 2 for r in team1_ratings) + sum(r.sigma ** 2 for r in team2_ratings)
    beta_sq = ts_env.beta ** 2
    denom = math.sqrt(len(team1_ratings) * beta_sq + len(team2_ratings) * beta_sq + sum_sigma_sq)
    return 0.5 * (1 + math.erf(delta_mu / (math.sqrt(2) * denom)))


app = Flask(__name__, static_folder='.', static_url_path='')


def load_data():
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"matches": [], "playerDetails": {}}


def save_data(data):
    with open(DATA_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def calculate_all_stats():
    data = load_data()
    matches = data.get("matches", [])
    player_details = data.get("playerDetails", {})

    player_names = set(p for m in matches for p in m['teamA'] + m['teamB'])

    # FIX: Week calculation now based on the fixed SEASON_START date
    if matches:
        season_start_ts = SEASON_START.timestamp() * 1000
        ONE_WEEK_MS = 7 * 24 * 60 * 60 * 1000
        for match in matches:
            if isinstance(match.get('ts'), (int, float)):
                time_diff = match['ts'] - season_start_ts
                match['week'] = math.floor(time_diff / ONE_WEEK_MS) + 1 if time_diff >= 0 else 0
            else:
                match['week'] = 1  # Fallback

    if not player_names:
        return {"leaderboard": [], "playerStats": {}, "matches": [], "playerDetails": {}, "history": {},
                "historicalLeaderboards": [], "superlatives": {}, "rivalries": [], "duos": [], "totalMatches": 0,
                "weeklyData": {}}

    ratings = {name: ts_env.create_rating() for name in player_names}
    history = {name: [{'mu': r.mu, 'sigma': r.sigma, 'match_id': -1}] for name, r in ratings.items()}
    historical_leaderboards = [{'leaderboard': [], 'match_id': -1}]
    last_played_index = {name: -1 for name in player_names}

    superlatives = {'upset': {'prob': 1}, 'dominant': {'margin': 0}, 'best_performance': {'skill_gain': 0}}
    rivalries, duos = {}, {}

    # NEW: Prepare data structure for weekly review
    weekly_data_aggregator = {}

    for match_index, match in enumerate(matches):
        team_a_ratings = {p: ratings[p] for p in match['teamA']}
        team_b_ratings = {p: ratings[p] for p in match['teamB']}

        # Superlative calculations
        win_prob_a = _win_probability(list(team_a_ratings.values()), list(team_b_ratings.values()))
        winner_prob = win_prob_a if match['scoreA'] > match['scoreB'] else 1 - win_prob_a
        if winner_prob < superlatives['upset']['prob']:
            superlatives['upset'] = {'prob': winner_prob, 'match_id': match_index}

        margin = abs(match['scoreA'] - match['scoreB'])
        if margin > superlatives['dominant']['margin']:
            superlatives['dominant'] = {'margin': margin, 'match_id': match_index}

        # Duo & Rivalry calculations
        for team, won in [(match['teamA'], match['scoreA'] > match['scoreB']),
                          (match['teamB'], match['scoreB'] > match['scoreA'])]:
            for p1, p2 in combinations(team, 2):
                key = tuple(sorted((p1, p2)));
                duos.setdefault(key, {'wins': 0, 'total': 0, 'players': key})['total'] += 1
                if won: duos[key]['wins'] += 1
        for p1 in match['teamA']:
            for p2 in match['teamB']:
                key = tuple(sorted((p1, p2)));
                rivalries.setdefault(key, {'p1_wins': 0, 'p2_wins': 0, 'total': 0, 'players': key})['total'] += 1
                if match['scoreA'] > match['scoreB']:
                    rivalries[key]['p1_wins' if p1 == key[0] else 'p2_wins'] += 1
                else:
                    rivalries[key]['p2_wins' if p1 == key[0] else 'p1_wins'] += 1

        ranks = [0, 1] if match['scoreA'] > match['scoreB'] else [1, 0]
        new_ratings = ts_env.rate([team_a_ratings, team_b_ratings], ranks=ranks)

        # NEW: Track weekly skill gains
        week_num = match.get('week', 0)
        if week_num > 0:
            weekly_data_aggregator.setdefault(week_num, {'skillGains': {}, 'matchesPlayed': {}})

        for team_ratings in new_ratings:
            for player_name, new_rating in team_ratings.items():
                old_rating = ratings[player_name]
                skill_gain = (new_rating.mu - 3 * new_rating.sigma) - (old_rating.mu - 3 * old_rating.sigma)
                if skill_gain > superlatives['best_performance']['skill_gain']:
                    superlatives['best_performance'] = {'skill_gain': skill_gain, 'player': player_name,
                                                        'match_id': match_index}

                if week_num > 0:
                    weekly_data_aggregator[week_num]['skillGains'].setdefault(player_name, 0)
                    weekly_data_aggregator[week_num]['skillGains'][player_name] += skill_gain
                    weekly_data_aggregator[week_num]['matchesPlayed'].setdefault(player_name, 0)
                    weekly_data_aggregator[week_num]['matchesPlayed'][player_name] += 1

                ratings[player_name] = new_rating
                history[player_name].append({'mu': new_rating.mu, 'sigma': new_rating.sigma, 'match_id': match_index})

        for p in match['teamA'] + match['teamB']: last_played_index[p] = match_index

        current_leaderboard_list = sorted([{'name': name, 'skill': r.mu - 3 * r.sigma} for name, r in ratings.items() if
                                           last_played_index[name] != -1], key=lambda x: x['skill'], reverse=True)
        prev_ranks = {p['name']: i for i, p in enumerate(historical_leaderboards[-1]['leaderboard'])}
        for i, player in enumerate(current_leaderboard_list): player['change'] = prev_ranks.get(player['name'], i) - i
        historical_leaderboards.append({'leaderboard': current_leaderboard_list, 'match_id': match_index})

        if week_num > 0: weekly_data_aggregator[week_num]['end_of_week_leaderboard_snapshot'] = current_leaderboard_list

    # --- Post-Calculation Analysis ---
    leaderboard = historical_leaderboards[-1]['leaderboard']
    player_stats = {}
    for name in player_names:
        wins, losses, win_streak, longest_streak = 0, 0, 0, 0
        for i, m in enumerate(matches):
            if name in m['teamA'] or name in m['teamB']:
                won = (name in m['teamA'] and m['scoreA'] > m['scoreB']) or (
                            name in m['teamB'] and m['scoreB'] > m['scoreA'])
                if won:
                    wins += 1; win_streak += 1
                else:
                    losses += 1; win_streak = 0
                longest_streak = max(longest_streak, win_streak)
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        player_stats[name] = {"name": name, "wins": wins, "losses": losses, "total": total,
                              "winRate": f"{win_rate:.1f}", "skill": ratings[name].mu - 3 * ratings[name].sigma,
                              "mu": ratings[name].mu, "sigma": ratings[name].sigma, "longestWinStreak": longest_streak,
                              "last_played": last_played_index[name]}

    for player in leaderboard:
        if player['name'] in player_stats: player_stats[player['name']]['change'] = player['change']

    # NEW: Finalize weekly review data
    final_weekly_data = {}
    for week_num, week_data in weekly_data_aggregator.items():
        top_performer = max(week_data['skillGains'], key=week_data['skillGains'].get) if week_data[
            'skillGains'] else None
        most_active = max(week_data['matchesPlayed'], key=week_data['matchesPlayed'].get) if week_data[
            'matchesPlayed'] else None
        final_weekly_data[week_num] = {
            "weekNumber": week_num,
            "leaderboard": week_data.get('end_of_week_leaderboard_snapshot', []),
            "topPerformer": {"name": top_performer,
                             "value": week_data['skillGains'].get(top_performer, 0)} if top_performer else None,
            "mostActive": {"name": most_active,
                           "value": week_data['matchesPlayed'].get(most_active, 0)} if most_active else None
        }

    final_duos = sorted([d for d in duos.values() if d['total'] >= 3],
                        key=lambda x: (x['wins'] / x['total'], x['total']), reverse=True)[:10]
    final_rivalries = sorted([r for r in rivalries.values() if r['total'] >= 3], key=lambda x: x['total'],
                             reverse=True)[:10]

    return {
        "leaderboard": leaderboard, "playerStats": player_stats, "matches": matches,
        "playerDetails": player_details, "history": history,
        "historicalLeaderboards": historical_leaderboards,
        "superlatives": superlatives, "rivalries": final_rivalries, "duos": final_duos,
        "totalMatches": len(matches), "weeklyData": final_weekly_data
    }


@app.route('/')
def serve_index(): return send_from_directory('.', 'index.html')


@app.route('/api/data')
def get_all_data(): return jsonify(calculate_all_stats())


@app.route('/api/add_match', methods=['POST'])
def add_match():
    payload = request.json
    if payload.get('secret') != ADMIN_SECRET: return jsonify({"error": "Invalid admin secret"}), 403
    data = load_data()
    data['matches'].append({k: payload[k] for k in ['teamA', 'teamB', 'scoreA', 'scoreB', 'ts']})
    for player in payload['teamA'] + payload['teamB']:
        if player not in data['playerDetails']: data['playerDetails'][player] = {"lastName": "", "abbrev": "",
                                                                                 "number": ""}
    save_data(data)
    return jsonify({"success": True})


@app.route('/api/update_player', methods=['POST'])
def update_player():
    payload = request.json
    if payload.get('secret') != ADMIN_SECRET: return jsonify({"error": "Invalid admin secret"}), 403
    old_name, new_name, details = payload['originalName'], payload['newName'], payload['details']
    data = load_data()
    if old_name != new_name:
        if new_name in data['playerDetails']: return jsonify({"error": f"Player '{new_name}' already exists."}), 409
        for match in data['matches']:
            match['teamA'] = [new_name if p == old_name else p for p in match['teamA']]
            match['teamB'] = [new_name if p == old_name else p for p in match['teamB']]
        if old_name in data['playerDetails']: data['playerDetails'][new_name] = data['playerDetails'].pop(old_name)
    data['playerDetails'][new_name] = details
    save_data(data)
    return jsonify({"success": True})


if __name__ == '__main__': app.run(debug=True, port=5001)

