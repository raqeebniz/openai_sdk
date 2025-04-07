# Dapr Agentic Cloud Ascent (DACA) Local Development

This repository contains a step-by-step guide to implementing the Dapr Agentic Cloud Ascent (DACA) design pattern for local development of cloud-native multi-agent systems. Each step builds upon the previous one, helping you understand and implement DACA's core concepts.

## Overview

DACA is a strategic framework for developing and deploying scalable, resilient, and cost-effective agentic AI systems. It leverages:
- OpenAI Agents SDK for agent logic
- Dapr for distributed resilience
- Progressive deployment strategy from local to planetary scale
- Event-driven architecture (EDA)
- Three-tier microservices architecture
- Stateless computing
- Scheduled computing (CronJobs)

## Project Structure

The project is organized into 30 steps, each focusing on a specific aspect of DACA implementation:

1. `01_fastapi` - Basic FastAPI setup
2. `02_openai_agents_with_fastapi` - Integrating OpenAI Agents with FastAPI
3. `03_microservices_with_fastapi` - Inter-service communication
4. `04_dapr_theory_and_cli` - Understanding Dapr concepts and CLI
5. `05_inter_microservices_communication_using_dapr` - Dapr-based communication
6. `06_docker_and_desktop` - Docker basics and Desktop setup
7. `07_containerzing_fast_api` - Containerizing FastAPI applications
8. `08_containerzing_dapr` - Containerizing Dapr applications
9. `09_inter_containerized_microservices_communication_using_dapr` - Dapr in containers
10. `10_docker_compose` - Docker Compose basics
11. `11_deploy_containerize_fastapi_and_dapr_with_docker_compose` - Deployment with Docker Compose
12. `12_redis` - Redis basics
13. `13_redis_using_dapr` - Redis integration with Dapr
14. `14_cockroachdb` - CockroachDB setup
15. `15_sqlmodel` - SQLModel ORM basics
16. `16_sqlmodel_with_fastapi` - SQLModel with FastAPI
17. `17_cockroachdb_sqlmodel_using_dapr` - Database integration with Dapr
18. `18_rabbitmq` - RabbitMQ setup
19. `19_scheduled_container_invocation` - Dapr Scheduler implementation
20. `20_async_inter_containerized_microservices_communication_using_rabbitmq_dapr_scheduled_containers` - Async communication
21. `21_human_in_the_loop_integration` - HITL implementation
22. `21_daca_testing` - Testing DACA components
23. `22_daca_chatbot_with_ui` - Chatbot implementation with UI
24. `23_custom_fastapi_dapr_gateway` - Custom gateway implementation
25. `24_dapr_workflows` - Dapr Workflows integration
26. `25_dapr_agents` - Dapr Agents implementation
27. `27_comprehensive_daca_multi_agent_example_with_workflows_agents_with_ui` - Complete example
28. `28_daca_dapr_observability` - Observability setup
29. `29_daca_dapr_security` - Security implementation
30. `30_daca_template` - Project template

## Prerequisites

- Python 3.8+
- Docker Desktop
- Dapr CLI
- OpenAI API key
- Basic understanding of:
  - Python
  - Docker
  - Microservices architecture
  - Event-driven systems




## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the Agents SDK
- Dapr team for the distributed application runtime
- The open-source community for various tools and libraries used in this project

## Resources

- [Dapr Documentation](https://docs.dapr.io/)
- [OpenAI Agents SDK](https://github.com/openai/openai-python)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Comprehensive DACA Guide](https://github.com/panaversity/learn-agentic-ai/blob/main/comprehensive_guide_daca.md) 
