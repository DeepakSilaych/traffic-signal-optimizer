import subprocess
import sys
import os
from pathlib import Path

DIR = Path(__file__).resolve().parent

NODES = {
    'E1':  (-300, 400), 'E2':  (-300, 200), 'E3':  (-300, 0),
    'E4':  (0, -200),   'E5':  (300, -200), 'E6':  (600, -200),
    'E7':  (900, -200),
    'E8':  (1100, 0),   'E9':  (1100, 200), 'E10': (1100, 400),
    'E11': (600, 600),   'E12': (200, 600),

    'J1':  (0, 400),     'J2':  (0, 200),    'J3':  (0, 0),
    'J4':  (200, 400),   'J5':  (200, 200),  'J6':  (200, 0),
    'J7':  (450, 400),   'J8':  (450, 200),  'J9':  (450, 0),
    'J10': (450, 130),
    'J11': (600, 400),   'J12': (600, 200),  'J13': (600, 0),
    'J14': (750, 350),   'J15': (750, 200),  'J16': (750, 0),
    'J17': (900, 400),   'J18': (900, 200),  'J19': (900, 0),
    'J20': (350, 300),
}

EDGES = [
    ('E1',  'J1',  'arterial'),  ('J1',  'E1',  'arterial'),
    ('E2',  'J2',  'arterial'),  ('J2',  'E2',  'arterial'),
    ('E3',  'J3',  'collector'), ('J3',  'E3',  'collector'),
    ('E4',  'J3',  'collector'), ('J3',  'E4',  'collector'),
    ('E5',  'J6',  'collector'), ('J6',  'E5',  'collector'),
    ('E6',  'J13', 'collector'), ('J13', 'E6',  'collector'),
    ('E7',  'J19', 'local'),     ('J19', 'E7',  'local'),
    ('E8',  'J19', 'collector'), ('J19', 'E8',  'collector'),
    ('E9',  'J18', 'collector'), ('J18', 'E9',  'collector'),
    ('E10', 'J17', 'arterial'),  ('J17', 'E10', 'arterial'),
    ('E11', 'J11', 'collector'), ('J11', 'E11', 'collector'),
    ('E12', 'J4',  'collector'), ('J4',  'E12', 'collector'),

    ('J1',  'J2',  'arterial'),  ('J2',  'J1',  'arterial'),
    ('J2',  'J3',  'arterial'),  ('J3',  'J2',  'arterial'),
    ('J1',  'J4',  'arterial'),  ('J4',  'J1',  'arterial'),
    ('J4',  'J7',  'arterial'),  ('J7',  'J4',  'arterial'),
    ('J7',  'J11', 'arterial'),  ('J11', 'J7',  'arterial'),
    ('J11', 'J17', 'arterial'),  ('J17', 'J11', 'arterial'),

    ('J2',  'J5',  'collector'), ('J5',  'J2',  'collector'),
    ('J3',  'J6',  'collector'), ('J6',  'J3',  'collector'),
    ('J4',  'J5',  'collector'), ('J5',  'J4',  'collector'),
    ('J5',  'J6',  'collector'), ('J6',  'J5',  'collector'),
    ('J5',  'J8',  'collector'), ('J8',  'J5',  'collector'),
    ('J6',  'J9',  'collector'), ('J9',  'J6',  'collector'),

    ('J7',  'J8',  'collector'), ('J8',  'J7',  'collector'),
    ('J8',  'J12', 'collector'), ('J12', 'J8',  'collector'),
    ('J9',  'J13', 'collector'), ('J13', 'J9',  'collector'),
    ('J9',  'J10', 'local'),     ('J10', 'J9',  'local'),

    ('J11', 'J12', 'collector'), ('J12', 'J11', 'collector'),
    ('J12', 'J13', 'collector'), ('J13', 'J12', 'collector'),
    ('J12', 'J15', 'collector'), ('J15', 'J12', 'collector'),
    ('J13', 'J16', 'local'),     ('J16', 'J13', 'local'),

    ('J14', 'J11', 'local'),     ('J11', 'J14', 'local'),
    ('J14', 'J15', 'local'),     ('J15', 'J14', 'local'),
    ('J15', 'J16', 'collector'), ('J16', 'J15', 'collector'),
    ('J15', 'J18', 'collector'), ('J18', 'J15', 'collector'),
    ('J16', 'J19', 'collector'), ('J19', 'J16', 'collector'),

    ('J17', 'J18', 'collector'), ('J18', 'J17', 'collector'),
    ('J18', 'J19', 'collector'), ('J19', 'J18', 'collector'),

    ('J5',  'J20', 'local'),     ('J20', 'J5',  'local'),
    ('J8',  'J20', 'local'),     ('J20', 'J8',  'local'),
]

TYPES = {
    'arterial':  {'numLanes': 3, 'speed': '13.89'},
    'collector': {'numLanes': 2, 'speed': '11.11'},
    'local':     {'numLanes': 1, 'speed': '8.33'},
}

ROUTES = [
    ('art_east',   'E1 J1 J4 J7 J11 J17 E10',                800),
    ('art_west',   'E10 J17 J11 J7 J4 J1 E1',                700),
    ('art_ns_1',   'E12 J4 J5 J6 E5',                         400),
    ('art_ns_2',   'E2 J2 J5 J8 J12 J15 J18 E9',             500),
    ('coll_1',     'E3 J3 J6 J9 J13 E6',                      350),
    ('coll_2',     'E11 J11 J12 J13 J16 J19 E8',              300),
    ('coll_3',     'E5 J6 J5 J2 J1 E1',                       250),
    ('coll_4',     'E9 J18 J17 J11 J7 J4 E12',                300),
    ('coll_diag',  'E4 J3 J2 J5 J20 J8 J12 J15 J18 E9',      200),
    ('local_1',    'E3 J3 J6 J9 J10',                         150),
    ('local_2',    'E7 J19 J16 J13 J9 J6 E5',                 150),
    ('local_3',    'E6 J13 J16 J15 J14',                      100),
    ('local_4',    'E2 J2 J3 J6 J9 J13 E6',                   200),
    ('local_5',    'E12 J4 J7 J8 J20 J5 J2 E2',              180),
    ('through_1',  'E1 J1 J4 J7 J8 J12 J15 J18 J19 E8',      250),
    ('through_2',  'E4 J3 J6 J9 J13 J16 J19 E7',             180),
]


def write_nodes():
    path = DIR / 'city_nodes.nod.xml'
    lines = ['<nodes>']
    dead_ends = {'J10', 'J14', 'J20'}
    for nid, (x, y) in NODES.items():
        ntype = 'traffic_light' if nid.startswith('J') and nid not in dead_ends else 'priority'
        lines.append(f'  <node id="{nid}" x="{x}" y="{y}" type="{ntype}"/>')
    lines.append('</nodes>')
    path.write_text('\n'.join(lines))
    return path


def write_types():
    path = DIR / 'city_types.typ.xml'
    lines = ['<types>']
    for tid, attrs in TYPES.items():
        lines.append(f'  <type id="{tid}" numLanes="{attrs["numLanes"]}" speed="{attrs["speed"]}"/>')
    lines.append('</types>')
    path.write_text('\n'.join(lines))
    return path


def write_edges():
    path = DIR / 'city_edges.edg.xml'
    lines = ['<edges>']
    for frm, to, etype in EDGES:
        eid = f'{frm}_{to}'
        lines.append(f'  <edge id="{eid}" from="{frm}" to="{to}" type="{etype}"/>')
    lines.append('</edges>')
    path.write_text('\n'.join(lines))
    return path


def run_netconvert(nod, edg, typ):
    out = DIR / 'city_network.net.xml'
    cmd = [
        'netconvert',
        '--node-files', str(nod),
        '--edge-files', str(edg),
        '--type-files', str(typ),
        '--output-file', str(out),
        '--no-turnarounds', 'true',
        '--junctions.join', 'false',
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return out


def _resolve_route_edges(node_seq_str):
    nodes = node_seq_str.split()
    edges = []
    for i in range(len(nodes) - 1):
        edges.append(f'{nodes[i]}_{nodes[i+1]}')
    return ' '.join(edges)


def write_routes():
    path = DIR / 'city_routes.rou.xml'
    lines = [
        '<routes>',
        '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.89"/>',
    ]

    for rid, node_seq, _ in ROUTES:
        edge_seq = _resolve_route_edges(node_seq)
        lines.append(f'  <route id="{rid}" edges="{edge_seq}"/>')

    for rid, _, vph in ROUTES:
        lines.append(
            f'  <flow id="f_{rid}" type="car" route="{rid}" '
            f'begin="0" end="3600" vehsPerHour="{vph}" departLane="best"/>'
        )

    lines.append('</routes>')
    path.write_text('\n'.join(lines))
    return path


def write_sumocfg():
    path = DIR / 'city_simulation.sumocfg'
    cfg = """<configuration>
    <input>
        <net-file value="city_network.net.xml"/>
        <route-files value="city_routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>"""
    path.write_text(cfg)
    return path


if __name__ == '__main__':
    nod = write_nodes()
    typ = write_types()
    edg = write_edges()
    print(f'Wrote {nod.name}, {typ.name}, {edg.name}')

    net = run_netconvert(nod, edg, typ)
    print(f'Generated {net.name}')

    rou = write_routes()
    print(f'Wrote {rou.name}')

    cfg = write_sumocfg()
    print(f'Wrote {cfg.name}')

    print('\nDone! Test with:')
    print(f'  sumo -c {cfg} --no-step-log --duration-log.statistics -e 100')
