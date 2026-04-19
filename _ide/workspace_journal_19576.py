# 2026-04-18T16:46:30.578220100
import vitis

client = vitis.create_client()
client.set_workspace(path="OrespawnStarkHawks")

platform = client.create_platform_component(name = "Bee_670",hw_design = "$COMPONENT_LOCATION/../design_1_wrapper.xsa",os = "standalone",cpu = "psu_cortexa53_0",domain_name = "standalone_psu_cortexa53_0",architecture = "64-bit",compiler = "gcc")

status = client.add_platform_repos(platform=["c:\GithubRepos\OrespawnStarkHawks\Bee_670"])

status = client.add_platform_repos(platform=["c:\GithubRepos\OrespawnStarkHawks\Bee_670"])

status = client.add_platform_repos(platform=["c:\GithubRepos\OrespawnStarkHawks\Bee_670"])

comp = client.create_app_component(name="app_component",platform = "$COMPONENT_LOCATION/../Bee_670/export/Bee_670/Bee_670.xpfm",domain = "standalone_psu_cortexa53_0")

platform = client.get_component(name="Bee_670")
status = platform.build()

comp = client.get_component(name="app_component")
comp.build()

status = platform.build()

comp.build()

status = platform.build()

comp.build()

vitis.dispose()

