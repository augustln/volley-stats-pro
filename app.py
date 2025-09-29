import json
import math
from flask import Flask, jsonify, request, send_from_directory
import trueskill
from itertools import combinations
from datetime import datetime, timezone
import os

# --- Configuration ---
ADMIN_SECRET = os.environ.get("VOLLEY_ADMIN_SECRET", "1243")
SEASON_START = datetime(2025, 9, 1, tzinfo=timezone.utc)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
JSON_FILE_PATH = os.path.join(APP_ROOT, 'volley-pro-matches.json')

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
    return ts_env.cdf(delta_mu / denom)


app = Flask(__name__, static_folder='.', static_url_path='')


def find_award_winners(data_dict, take_highest=True):
    """Helper to find winners, handling ties."""
    if not data_dict:
        return {'names': [], 'value': 0}

    sorted_players = sorted(data_dict.items(), key=lambda item: item[1], reverse=take_highest)
    if not sorted_players:
        return {'names': [], 'value': 0}

    top_value = sorted_players[0][1]
    winners = [name for name, value in sorted_players if abs(value - top_value) < 1e-9]
    return {'names': winners, 'value': top_value}


def calculate_all_stats(data):
    """
    The main calculation engine. Processes all matches to generate stats.
    """
    matches = data.get("matches", [])
    player_details = data.get("playerDetails", {})

    if not matches:
        return {"leaderboard": [], "playerStats": {}, "matches": [], "playerDetails": {}, "history": {},
                "superlatives": {}, "rivalries": [], "duos": [], "totalMatches": 0, "weeklyData": {}}

    season_start_ts = SEASON_START.timestamp() * 1000
    ONE_WEEK_MS = 7 * 24 * 60 * 60 * 1000
    for match in matches:
        if isinstance(match.get('ts'), (int, float)):
            time_diff = match['ts'] - season_start_ts
            match['week'] = math.floor(time_diff / ONE_WEEK_MS) + 1 if time_diff >= 0 else 0
        else:
            match['week'] = 1

    player_names = set(p for m in matches for p in m['teamA'] + m['teamB'])

    ratings = {name: ts_env.create_rating() for name in player_names}
    history = {name: [{'mu': r.mu, 'sigma': r.sigma, 'match_id': -1}] for name, r in ratings.items()}
    last_played_index = {name: -1 for name in player_names}
    leaderboard_before_last = []

    superlatives = {
        'upset': {'prob': 1}, 'dominant': {'margin': 0},
        'best_performance': {'skill_gain': 0}, 'highCaliber': {'avgSkill': 0}
    }
    rivalries, duos = {}, {}
    weekly_data_aggregator = {}

    for match_index, match in enumerate(matches):
        team_a_ratings_map = {p: ratings[p] for p in match['teamA']}
        team_b_ratings_map = {p: ratings[p] for p in match['teamB']}

        win_prob_a = _win_probability(list(team_a_ratings_map.values()), list(team_b_ratings_map.values()))
        winner_prob = win_prob_a if match['scoreA'] > match['scoreB'] else 1 - win_prob_a
        if winner_prob < superlatives['upset']['prob']:
            superlatives['upset'] = {'prob': winner_prob, 'match_id': match_index}

        margin = abs(match['scoreA'] - match['scoreB'])
        if margin > superlatives['dominant']['margin']:
            superlatives['dominant'] = {'margin': margin, 'match_id': match_index}

        avg_skill_a = sum(r.mu - 3 * r.sigma for r in team_a_ratings_map.values()) / len(
            team_a_ratings_map) if team_a_ratings_map else 0
        avg_skill_b = sum(r.mu - 3 * r.sigma for r in team_b_ratings_map.values()) / len(
            team_b_ratings_map) if team_b_ratings_map else 0
        avg_match_skill = (avg_skill_a + avg_skill_b) / 2
        if avg_match_skill > superlatives['highCaliber']['avgSkill']:
            superlatives['highCaliber'] = {'avgSkill': avg_match_skill, 'match_id': match_index}

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

        ranks = [0, 1] if match['scoreA'] > match['scoreB'] else [1, 0]
        new_ratings_map = ts_env.rate([team_a_ratings_map, team_b_ratings_map], ranks=ranks)
        new_ratings = {p: r for team in new_ratings_map for p, r in team.items()}

        week_num = match.get('week', 0)
        if week_num > 0:
            w_agg = weekly_data_aggregator.setdefault(week_num,
                                                      {'skillGains': {}, 'matchesPlayed': {}, 'clutchWins': {},
                                                       'clutchAttempts': {}, 'pointDifferential': {}, 'wins': {},
                                                       'duoWins': {}, 'upset': {'prob': 1}, 'dominant': {'margin': 0}})
            if winner_prob < w_agg['upset']['prob']: w_agg['upset'] = {'prob': winner_prob, 'match_id': match_index}
            if margin > w_agg['dominant']['margin']: w_agg['dominant'] = {'margin': margin, 'match_id': match_index}

        for player_name, new_rating in new_ratings.items():
            old_rating = ratings[player_name]
            skill_gain = (new_rating.mu - 3 * new_rating.sigma) - (old_rating.mu - 3 * old_rating.sigma)
            if skill_gain > superlatives['best_performance']['skill_gain']:
                superlatives['best_performance'] = {'skill_gain': skill_gain, 'player': player_name,
                                                    'match_id': match_index}

            ratings[player_name] = new_rating
            history[player_name].append({'mu': new_rating.mu, 'sigma': new_rating.sigma, 'match_id': match_index})

            if week_num > 0:
                is_winner = (player_name in match['teamA'] and match['scoreA'] > match['scoreB']) or (
                            player_name in match['teamB'] and match['scoreB'] > match['scoreA'])
                point_diff = (match['scoreA'] - match['scoreB']) if player_name in match['teamA'] else (
                            match['scoreB'] - match['scoreA'])
                w_agg['skillGains'][player_name] = w_agg['skillGains'].get(player_name, 0) + skill_gain
                w_agg['matchesPlayed'][player_name] = w_agg['matchesPlayed'].get(player_name, 0) + 1
                w_agg['pointDifferential'][player_name] = w_agg['pointDifferential'].get(player_name, 0) + point_diff
                if is_winner: w_agg['wins'][player_name] = w_agg['wins'].get(player_name, 0) + 1
                if abs(match['scoreA'] - match['scoreB']) == 1:
                    w_agg['clutchAttempts'][player_name] = w_agg['clutchAttempts'].get(player_name, 0) + 1
                    if is_winner: w_agg['clutchWins'][player_name] = w_agg['clutchWins'].get(player_name, 0) + 1

        if week_num > 0:
            for team, won in [(match['teamA'], match['scoreA'] > match['scoreB']),
                              (match['teamB'], match['scoreB'] > match['scoreA'])]:
                if won:
                    for p1, p2 in combinations(team, 2):
                        key = tuple(sorted((p1, p2)))
                        w_agg['duoWins'][key] = w_agg['duoWins'].get(key, 0) + 1

        for p in match['teamA'] + match['teamB']: last_played_index[p] = match_index

        # --- FEATURE: Capture leaderboard state before the final match ---
        if len(matches) > 1 and match_index == len(matches) - 2:
            leaderboard_before_last = sorted(
                [{'name': name, 'skill': r.mu - 3 * r.sigma, 'mu': r.mu, 'sigma': r.sigma} for name, r in
                 ratings.items()],
                key=lambda x: x['skill'], reverse=True)

    for week_num, w_data in weekly_data_aggregator.items():
        snapshot = []
        active_players = w_data.get('matchesPlayed', {}).keys()
        for player_name in active_players:
            weekly_match_ids = [h['match_id'] for h in history[player_name] if
                                h['match_id'] != -1 and matches[h['match_id']].get('week') == week_num]
            if not weekly_match_ids: continue
            last_match_id_for_week = max(weekly_match_ids)
            rating_at_week_end = next((h for h in history[player_name] if h['match_id'] == last_match_id_for_week),
                                      None)
            if rating_at_week_end:
                snapshot.append(
                    {'name': player_name, 'skill': rating_at_week_end['mu'] - 3 * rating_at_week_end['sigma'],
                     'mu': rating_at_week_end['mu'], 'sigma': rating_at_week_end['sigma']})
        w_data['end_of_week_leaderboard_snapshot'] = sorted(snapshot, key=lambda x: x['skill'], reverse=True)

    final_leaderboard = sorted(
        [{'name': name, 'skill': r.mu - 3 * r.sigma} for name, r in ratings.items()],
        key=lambda x: x['skill'], reverse=True)

    player_stats = {}
    for name in player_names:
        p_history = history[name]
        skill_history = [h['mu'] - 3 * h['sigma'] for h in p_history]
        wins, losses, win_streak, longest_streak = 0, 0, 0, 0
        clutch_wins, clutch_attempts = 0, 0
        current_streak, streak_type = 0, ''

        if len(p_history) > 1:
            streak_type = 'W' if p_history[-1]['mu'] > p_history[-2]['mu'] else 'L'
            for i in range(len(p_history) - 1, 0, -1):
                is_win = p_history[i]['mu'] > p_history[i - 1]['mu']
                if (is_win and streak_type == 'W') or (not is_win and streak_type == 'L'):
                    current_streak += 1
                else:
                    break

        for m in matches:
            if name in m['teamA'] or name in m['teamB']:
                won = (name in m['teamA'] and m['scoreA'] > m['scoreB']) or (
                            name in m['teamB'] and m['scoreB'] > m['scoreA'])
                if won:
                    wins += 1; win_streak += 1
                else:
                    losses += 1; win_streak = 0
                longest_streak = max(longest_streak, win_streak)
                if abs(m['scoreA'] - m['scoreB']) == 1:
                    clutch_attempts += 1
                    if won: clutch_wins += 1

        total_games = wins + losses
        player_stats[name] = {"name": name, "wins": wins, "losses": losses, "total": total_games,
                              "winRate": f"{(wins / total_games * 100) if total_games > 0 else 0:.1f}",
                              "skill": ratings[name].mu - 3 * ratings[name].sigma,
                              "peakSkill": max(skill_history) if skill_history else 0, "mu": ratings[name].mu,
                              "sigma": ratings[name].sigma, "longestWinStreak": longest_streak,
                              "currentStreak": {'count': current_streak, 'type': streak_type},
                              "clutchRate": (clutch_wins / clutch_attempts * 100) if clutch_attempts > 0 else 0,
                              "skillChangeLast10": (skill_history[-1] - skill_history[-11]) if len(
                                  skill_history) > 10 else (
                                  skill_history[-1] - skill_history[0] if len(skill_history) > 1 else 0),
                              "last_played": last_played_index[name]}

    # First pass: build the basic weekly data
    final_weekly_data = {}
    for week_num, w_data in weekly_data_aggregator.items():
        active_players = set(w_data['matchesPlayed'].keys())
        clutch_performances = {
            p: (w_data['clutchWins'].get(p, 0) / w_data['clutchAttempts'].get(p, 1), w_data['clutchAttempts'].get(p, 0))
            for p in active_players if w_data['clutchAttempts'].get(p, 0) >= 3}
        best_clutcher_val = max(clutch_performances.values()) if clutch_performances else (0, 0)
        clutcher_names = [p for p, v in clutch_performances.items() if v == best_clutcher_val]
        best_duo_key = max(w_data['duoWins'], key=w_data['duoWins'].get, default=None)
        best_duo_wins = w_data['duoWins'].get(best_duo_key, 0)
        best_duos = [list(k) for k, v in w_data['duoWins'].items() if v == best_duo_wins] if best_duo_wins > 0 else []

        final_weekly_data[week_num] = {
            "weekNumber": week_num, "leaderboard": w_data.get('end_of_week_leaderboard_snapshot', []),
            "topPerformer": find_award_winners(w_data['skillGains']),
            "mostActive": find_award_winners(w_data['matchesPlayed']),
            "bestPointDiff": find_award_winners(w_data['pointDifferential']),
            "mostWins": find_award_winners(w_data.get('wins', {})),
            "weeklyClutcher": {"names": clutcher_names, "value": best_clutcher_val[0] * 100,
                               "attempts": best_clutcher_val[1]},
            "bestDuo": {"players_list": best_duos, "wins": best_duo_wins},
            "biggestBottler": find_award_winners(w_data['skillGains'], take_highest=False),
            "worstPointDiff": find_award_winners(w_data['pointDifferential'], take_highest=False),
            "upset": w_data['upset'], "dominant": w_data['dominant']
        }

    # --- FEATURE: Second pass to add start-of-week leaderboards for comparison ---
    for week_num in sorted(final_weekly_data.keys()):
        if week_num > 1 and (week_num - 1) in final_weekly_data:
            final_weekly_data[week_num]['start_of_week_leaderboard'] = final_weekly_data[week_num - 1]['leaderboard']
        else:
            final_weekly_data[week_num]['start_of_week_leaderboard'] = []

    final_duos = sorted([d for d in duos.values() if d['total'] >= 5],
                        key=lambda x: (x['wins'] / x['total'], x['total']), reverse=True)[:10]
    final_rivalries = sorted([r for r in rivalries.values() if r['total'] >= 5], key=lambda x: x['total'],
                             reverse=True)[:10]

    return {
        "leaderboard": final_leaderboard, "leaderboard_before_last": leaderboard_before_last,
        "playerStats": player_stats, "matches": matches, "playerDetails": player_details,
        "history": history, "superlatives": superlatives, "rivalries": final_rivalries,
        "duos": final_duos, "totalMatches": len(matches), "weeklyData": final_weekly_data
    }


# --- API Routes ---
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


@app.route('/api/get_initial_data')
def get_initial_data():
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'matches' not in data: data['matches'] = []
        if 'playerDetails' not in data: data['playerDetails'] = {}
        return jsonify(data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        app.logger.error(f"Could not load data from {JSON_FILE_PATH}: {e}")
        return jsonify({"matches": [], "playerDetails": {}})


@app.route('/api/process_data', methods=['POST'])
def process_data_api():
    data = request.json
    if not data: return jsonify({"error": "No data provided"}), 400
    return jsonify(calculate_all_stats(data))


@app.route('/api/head2head', methods=['POST'])
def head2head_api():
    data = request.json
    p1, p2, matches = data.get('player1'), data.get('player2'), data.get('matches', [])
    if not all([p1, p2, matches]): return jsonify({"error": "Missing data"}), 400
    p1_wins, p2_wins, total = 0, 0, 0
    for m in matches:
        p1_on_a, p1_on_b = p1 in m['teamA'], p1 in m['teamB']
        p2_on_a, p2_on_b = p2 in m['teamA'], p2 in m['teamB']
        if (p1_on_a and p2_on_b) or (p1_on_b and p2_on_a):
            total += 1
            a_won = m['scoreA'] > m['scoreB']
            if (p1_on_a and a_won) or (p1_on_b and not a_won):
                p1_wins += 1
            else:
                p2_wins += 1
    return jsonify({"player1_wins": p1_wins, "player2_wins": p2_wins, "total_matches": total})


@app.route('/api/match_details/<int:match_id>', methods=['POST'])
def match_details_api(match_id):
    data = request.json
    matches, history = data.get('matches', []), data.get('history', {})
    if match_id >= len(matches) or not history: return jsonify({"error": "Invalid data"}), 400
    match = matches[match_id]

    def get_pre_match_rating(p_name):
        p_hist = history.get(p_name, [])
        for i in range(len(p_hist) - 1, -1, -1):
            if p_hist[i]['match_id'] < match_id: return ts_env.create_rating(mu=p_hist[i]['mu'],
                                                                             sigma=p_hist[i]['sigma'])
        return ts_env.create_rating()

    team_a_ratings = [get_pre_match_rating(p) for p in match['teamA']]
    team_b_ratings = [get_pre_match_rating(p) for p in match['teamB']]
    win_prob_a = _win_probability(team_a_ratings, team_b_ratings)

    def get_skill_change(p_name):
        p_hist = history.get(p_name, [])
        post_entry, pre_entry = None, None
        for i, entry in enumerate(p_hist):
            if entry['match_id'] == match_id:
                post_entry = entry
                if i > 0: pre_entry = p_hist[i - 1]
                break
        if post_entry and pre_entry:
            pre_skill, post_skill = pre_entry['mu'] - 3 * pre_entry['sigma'], post_entry['mu'] - 3 * post_entry['sigma']
            return {"name": p_name, "preSkill": f"{pre_skill:.2f}", "postSkill": f"{post_skill:.2f}",
                    "change": f"{post_skill - pre_skill:+.2f}"}
        return {"name": p_name, "preSkill": "N/A", "postSkill": "N/A", "change": "N/A"}

    team_a_changes = [get_skill_change(p) for p in match['teamA']]
    team_b_changes = [get_skill_change(p) for p in match['teamB']]

    return jsonify({"win_prob_a": win_prob_a, "team_a_changes": team_a_changes, "team_b_changes": team_b_changes})


if __name__ == '__main__':
    app.run(debug=True, port=5001)

