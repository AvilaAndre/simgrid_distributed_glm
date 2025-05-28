from simgrid import Host, NetZone


def star(zone: NetZone, hosts: list[Host], suffix: str):
    center_host = zone.add_host(f"CenterHost{suffix}", 25e6)

    for i in range(len(hosts)):
        host: Host = hosts[i]
        link = zone.add_split_duplex_link(f"link_{host.name}_center", 1e6)
        # FIXME: With lower latency (replacing the 4 with a 6) glm fails
        link.set_latency(10e-4).seal()
        zone.add_route(host, center_host, [link])

    for i in range(len(hosts)):
        for j in range(i + 1, len(hosts)):
            hostA, hostB = hosts[i], hosts[j]
            links = hostA.route_to(center_host)[0] + hostB.route_to(center_host)[0]
            zone.add_route(hostA, hostB, links)

