set(SOURCES
    vsgplanarreflection.cpp
)

add_executable(vsgplanarreflection ${SOURCES})

target_link_libraries(vsgplanarreflection vsg::vsg)

if (vsgXchange_FOUND)
    target_compile_definitions(vsgplanarreflection PRIVATE vsgXchange_FOUND)
    target_link_libraries(vsgplanarreflection vsgXchange::vsgXchange)
endif()

install(TARGETS vsgplanarreflection RUNTIME DESTINATION bin)
